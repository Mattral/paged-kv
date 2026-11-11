//! First-come, first-served continuous batching.
//!
//! Waiting sequences sit in arrival order. Admission takes the oldest
//! sequence whose prompt fits in the pool and stops when the batch is full.
//! It does not skip a prompt that does not fit, so a large head-of-line
//! request holds up smaller ones behind it.
//!
//! A decode step appends one token to every running sequence, or to none of
//! them if the pool cannot cover the whole batch. Finished sequences keep
//! their batch slot until [`Scheduler::retire`] releases their blocks.
//!
//! The scheduler does not share prefixes across sequences. Prompt tokens are
//! reserved in full at admission. There is no chunked prefill and no
//! preemption.

use std::collections::VecDeque;

use crate::sequence::{Phase, Sequence, SequenceId};
use crate::{AllocError, BlockPool};

/// One decode step over the running batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeStep {
    /// Sequences that received one new token, in batch order.
    pub decoded: Vec<SequenceId>,
    /// Sequences whose token budget was spent by this step, in batch order.
    pub finished: Vec<SequenceId>,
}

/// Queue of sequences sharing one [`BlockPool`].
#[derive(Debug)]
pub struct Scheduler {
    pool: BlockPool,
    max_batch: usize,
    next_id: u64,
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
}

impl Scheduler {
    /// `max_batch` is how many sequences may hold blocks at once.
    ///
    /// # Panics
    /// Panics if `max_batch` is zero.
    pub fn new(pool: BlockPool, max_batch: usize) -> Self {
        assert!(max_batch > 0, "max_batch must be non-zero");
        Self {
            pool,
            max_batch,
            next_id: 0,
            waiting: VecDeque::new(),
            running: Vec::new(),
        }
    }

    /// Queue a request. This does not reserve blocks.
    ///
    /// Ids increase from zero in arrival order. That order is the admission
    /// order.
    pub fn add(&mut self, prompt_len: u32, max_new_tokens: u32) -> SequenceId {
        let id = SequenceId::new(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("sequence id space exhausted");
        self.waiting
            .push_back(Sequence::new(id, prompt_len, max_new_tokens));
        id
    }

    /// Move waiting sequences into the batch, oldest first.
    ///
    /// Stops when the batch is full, the queue is empty, or the oldest
    /// waiting prompt does not fit. Earlier sequences admitted by this call
    /// stay running. A prompt that does not fit is left at the head of the
    /// queue.
    pub fn admit(&mut self) -> Result<Vec<SequenceId>, AllocError> {
        let mut admitted = Vec::new();
        let block_size = self.pool.block_size();
        while self.running.len() < self.max_batch {
            let Some(head) = self.waiting.front() else {
                break;
            };
            let need = BlockPool::blocks_needed(head.prompt_len(), block_size);
            if self.pool.free_count() < need {
                break;
            }
            let mut seq = self.waiting.pop_front().expect("front was observed");
            if let Err(err) = seq.allocate_prompt(&mut self.pool) {
                self.waiting.push_front(seq);
                return Err(err);
            }
            admitted.push(seq.id());
            self.running.push(seq);
        }
        Ok(admitted)
    }

    /// Append one generated token to every decoding sequence.
    ///
    /// If the pool cannot hold every new block the batch needs, no sequence
    /// advances. Finished sequences are skipped and stay in the batch.
    pub fn step(&mut self) -> Result<DecodeStep, AllocError> {
        let block_size = self.pool.block_size();
        let mut extra = 0u64;
        for seq in &self.running {
            if seq.phase() != Phase::Decoding {
                continue;
            }
            extra += u64::from(blocks_for_one_more_token(seq.len(), block_size));
        }
        if u64::from(self.pool.free_count()) < extra {
            return Err(AllocError::Exhausted);
        }

        let mut decoded = Vec::new();
        let mut finished = Vec::new();
        for seq in &mut self.running {
            if seq.phase() != Phase::Decoding {
                continue;
            }
            let id = seq.id();
            let done = seq.push_token(&mut self.pool)?;
            decoded.push(id);
            if done {
                finished.push(id);
            }
        }
        Ok(DecodeStep { decoded, finished })
    }

    /// Release blocks for every finished sequence and remove them.
    ///
    /// Sequences that are still decoding stay in the batch. If a release
    /// fails, later finished sequences are still released, and the first
    /// error is returned.
    pub fn retire(&mut self) -> Result<Vec<SequenceId>, AllocError> {
        let mut kept = Vec::with_capacity(self.running.len());
        let mut retired = Vec::new();
        let mut error = None;
        for seq in self.running.drain(..) {
            if seq.phase() != Phase::Finished {
                kept.push(seq);
                continue;
            }
            retired.push(seq.id());
            if let Err(err) = seq.release(&mut self.pool) {
                if error.is_none() {
                    error = Some(err);
                }
            }
        }
        self.running = kept;
        match error {
            Some(err) => Err(err),
            None => Ok(retired),
        }
    }

    /// Sequences waiting to be admitted.
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Oldest sequence that has not been admitted.
    pub fn next_waiting(&self) -> Option<&Sequence> {
        self.waiting.front()
    }

    /// Sequences that currently hold blocks, in admission order.
    pub fn running(&self) -> &[Sequence] {
        &self.running
    }

    /// Pool the batch allocates from.
    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    /// `true` when nothing is waiting or running.
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Maximum number of sequences that may hold blocks at once.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }
}

fn blocks_for_one_more_token(len: u32, block_size: u32) -> u32 {
    let next = len.checked_add(1).expect("sequence length fits in u32");
    BlockPool::blocks_needed(next, block_size) - BlockPool::blocks_needed(len, block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler(blocks: u32, block_size: u32, max_batch: usize) -> Scheduler {
        Scheduler::new(BlockPool::new(blocks, block_size), max_batch)
    }

    #[test]
    fn admission_follows_arrival_order_and_stops_at_the_batch_size() {
        let mut sched = scheduler(8, 4, 2);
        let first = sched.add(4, 1);
        let second = sched.add(4, 1);
        let third = sched.add(4, 1);

        let admitted = sched.admit().unwrap();
        assert_eq!(admitted, vec![first, second]);
        assert_eq!(sched.running().len(), 2);
        assert_eq!(sched.waiting_len(), 1);
        assert_eq!(sched.next_waiting().unwrap().id(), third);
        assert_eq!(sched.running()[0].prompt_len(), 4);
        assert_eq!(sched.running()[0].phase(), Phase::Decoding);
    }

    #[test]
    fn a_prompt_that_does_not_fit_blocks_smaller_requests_behind_it() {
        let mut sched = scheduler(2, 4, 4);
        let large = sched.add(12, 1);
        let small = sched.add(4, 1);

        let admitted = sched.admit().unwrap();
        assert!(admitted.is_empty());
        assert!(sched.running().is_empty());
        assert_eq!(sched.waiting_len(), 2);
        assert_eq!(sched.next_waiting().unwrap().id(), large);
        assert_eq!(sched.pool().free_count(), 2);
        let _ = small;
    }

    #[test]
    fn a_finished_sequence_keeps_its_slot_until_retire() {
        let mut sched = scheduler(4, 4, 1);
        let first = sched.add(4, 0);
        let second = sched.add(4, 0);

        assert_eq!(sched.admit().unwrap(), vec![first]);
        assert_eq!(sched.running()[0].phase(), Phase::Finished);
        assert!(sched.admit().unwrap().is_empty());
        assert_eq!(sched.waiting_len(), 1);

        assert_eq!(sched.retire().unwrap(), vec![first]);
        assert_eq!(sched.admit().unwrap(), vec![second]);
        assert_eq!(sched.retire().unwrap(), vec![second]);
        assert!(sched.is_idle());
        assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    }

    #[test]
    fn step_is_all_or_nothing_when_the_batch_cannot_grow() {
        let mut sched = scheduler(2, 4, 2);
        let first = sched.add(4, 2);
        let second = sched.add(4, 2);
        sched.admit().unwrap();
        assert_eq!(sched.pool().free_count(), 0);

        assert_eq!(sched.step().unwrap_err(), AllocError::Exhausted);
        assert!(sched.running().iter().all(|seq| seq.generated() == 0));
        assert_eq!(sched.running()[0].id(), first);
        assert_eq!(sched.running()[1].id(), second);
        assert_eq!(sched.pool().free_count(), 0);
    }

    #[test]
    fn step_then_retire_frees_blocks_and_admits_the_next_sequence() {
        let mut sched = scheduler(8, 4, 1);
        let first = sched.add(4, 1);
        let second = sched.add(8, 1);

        assert_eq!(sched.admit().unwrap(), vec![first]);
        let step = sched.step().unwrap();
        assert_eq!(step.decoded, vec![first]);
        assert_eq!(step.finished, vec![first]);
        assert_eq!(sched.running()[0].generated(), 1);
        assert_eq!(sched.running()[0].len(), 5);

        assert_eq!(sched.retire().unwrap(), vec![first]);
        assert_eq!(sched.pool().free_count(), sched.pool().capacity());
        assert_eq!(sched.admit().unwrap(), vec![second]);
        assert_eq!(sched.running()[0].len(), 8);

        let step = sched.step().unwrap();
        assert_eq!(step.finished, vec![second]);
        sched.retire().unwrap();
        assert!(sched.is_idle());
        assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    }
}
