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
//! reserved in full at admission. There is no chunked prefill. Preemption is
//! a caller hook: [`Scheduler::preempt`] removes the newest decoding
//! sequence. Admission never preempts on its own.
//!
//! A pool from [`BlockPool::new`] only tracks block ids. A pool from
//! [`BlockPool::with_storage`] writes K and V from [`TinyModel`](crate::TinyModel)
//! and chooses the next token with that block. Callers that did not pass
//! token ids get `1, 2, 3, ...` for the prompt. An id outside the byte
//! vocabulary is rejected.

use std::collections::{HashMap, VecDeque};

use crate::sequence::{Phase, Sequence, SequenceId};
use crate::tokenizer::{EOS, VOCAB_SIZE};
use crate::{AllocError, BlockPool, TinyModel};

/// Token ids and the last attention scores for sequences on a storage pool.
#[derive(Debug)]
struct KvTrack {
    head_dim: u32,
    model: TinyModel,
    /// Prompt ids from [`Scheduler::add_tokens`]. Absent means `1..=prompt_len`.
    prompts: HashMap<SequenceId, Vec<u32>>,
    /// Ids written into slots, prompt first, then generated tokens.
    written: HashMap<SequenceId, Vec<u32>>,
    /// Next id chosen by the block, waiting for the following step.
    pending: HashMap<SequenceId, u32>,
    /// Last softmax weights for a sequence that holds blocks.
    scores: HashMap<SequenceId, Vec<f32>>,
    /// Last mixed V vector for that same step.
    context: HashMap<SequenceId, Vec<f32>>,
}

/// What happens to a sequence taken out of the batch by [`Scheduler::preempt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptPolicy {
    /// Free its blocks and discard generated tokens.
    ///
    /// It waits again and will be prefilled from scratch. This is the policy
    /// that returns memory to the pool.
    Recompute,
    /// Keep its blocks and its generated tokens.
    ///
    /// It waits outside the batch. The next time it is admitted, decode
    /// continues. This frees a batch slot and does not free pool memory.
    Pause,
}

/// [`Scheduler::preempt`] could not finish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreemptError {
    /// The batch has no sequence that is still decoding.
    NothingRunning,
    /// Recompute failed while releasing blocks.
    Alloc(AllocError),
}

impl std::fmt::Display for PreemptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreemptError::NothingRunning => write!(f, "no decoding sequence to preempt"),
            PreemptError::Alloc(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for PreemptError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PreemptError::NothingRunning => None,
            PreemptError::Alloc(err) => Some(err),
        }
    }
}

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
    /// Present only when the pool stores vectors of an even width.
    kv: Option<KvTrack>,
}

impl Scheduler {
    /// `max_batch` is how many sequences may hold blocks at once.
    ///
    /// # Panics
    /// Panics if `max_batch` is zero.
    pub fn new(pool: BlockPool, max_batch: usize) -> Self {
        assert!(max_batch > 0, "max_batch must be non-zero");
        let kv = pool.floats_per_token().and_then(|width| {
            let head_dim = width / 2;
            if head_dim == 0 || head_dim.saturating_mul(2) != width {
                None
            } else {
                Some(KvTrack {
                    head_dim,
                    model: TinyModel::new(head_dim),
                    prompts: HashMap::new(),
                    written: HashMap::new(),
                    scores: HashMap::new(),
                    context: HashMap::new(),
                    pending: HashMap::new(),
                })
            }
        });
        Self {
            pool,
            max_batch,
            next_id: 0,
            waiting: VecDeque::new(),
            running: Vec::new(),
            kv,
        }
    }

    /// `Some(head_dim)` when this step must write KV.
    ///
    /// A metadata pool is `Ok(None)`. Storage whose width is not two equal
    /// halves is [`AllocError::KvWidth`].
    fn storage_head(&self) -> Result<Option<u32>, AllocError> {
        let Some(width) = self.pool.floats_per_token() else {
            return Ok(None);
        };
        match &self.kv {
            Some(kv) => Ok(Some(kv.head_dim)),
            None => Err(AllocError::KvWidth {
                floats_per_token: width,
                head_dim: width / 2,
            }),
        }
    }

    fn prompt_ids(&self, id: SequenceId, prompt_len: u32) -> Vec<u32> {
        if let Some(ids) = self.kv.as_ref().and_then(|kv| kv.prompts.get(&id)) {
            return ids.clone();
        }
        (1..=prompt_len).collect()
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

    /// Queue a prompt made of these token ids. This does not reserve blocks.
    ///
    /// The prompt length is `token_ids.len()`. On a storage pool, admission
    /// writes these ids. On a metadata pool they only supply that length.
    ///
    /// # Panics
    /// Panics if `token_ids.len()` does not fit in `u32`.
    pub fn add_tokens(&mut self, token_ids: &[u32], max_new_tokens: u32) -> SequenceId {
        let prompt_len = u32::try_from(token_ids.len()).expect("prompt length fits in u32");
        let id = self.add(prompt_len, max_new_tokens);
        if let Some(kv) = self.kv.as_mut() {
            kv.prompts.insert(id, token_ids.to_vec());
        }
        id
    }

    /// Token ids written for `id`, prompt first.
    ///
    /// `None` when this pool stores no vectors, or the sequence has not been
    /// admitted since it was last recomputed.
    pub fn token_ids(&self, id: SequenceId) -> Option<&[u32]> {
        self.kv.as_ref()?.written.get(&id).map(Vec::as_slice)
    }

    /// Last attention scores for `id`.
    ///
    /// Softmax weights, one per token currently in the sequence. `None` on
    /// the same conditions as [`Scheduler::token_ids`]. An empty prompt has
    /// an empty score list until the first generated token is written.
    pub fn scores(&self, id: SequenceId) -> Option<&[f32]> {
        self.kv.as_ref()?.scores.get(&id).map(Vec::as_slice)
    }

    /// Last attention context for `id`: the scores applied to stored V.
    ///
    /// `None` on the same conditions as [`Scheduler::scores`].
    pub fn context(&self, id: SequenceId) -> Option<&[f32]> {
        self.kv.as_ref()?.context.get(&id).map(Vec::as_slice)
    }

    /// Move waiting sequences into the batch, oldest first.
    ///
    /// Stops when the batch is full, the queue is empty, or the oldest
    /// waiting prompt does not fit. Earlier sequences admitted by this call
    /// stay running. A prompt that does not fit is left at the head of the
    /// queue.
    ///
    /// A paused sequence already holds its blocks. Admission resumes it
    /// without asking the pool for more. On a storage pool, a fresh admission
    /// writes the prompt's K and V from the fixed block and runs that block.
    /// A resumed sequence is not rewritten.
    pub fn admit(&mut self) -> Result<Vec<SequenceId>, AllocError> {
        let head_dim = self.storage_head()?;
        let mut admitted = Vec::new();
        let block_size = self.pool.block_size();
        while self.running.len() < self.max_batch {
            let Some(head) = self.waiting.front() else {
                break;
            };
            if head.phase() != Phase::Paused {
                let need = BlockPool::blocks_needed(head.prompt_len(), block_size);
                if self.pool.free_count() < need {
                    break;
                }
            }
            let mut seq = self.waiting.pop_front().expect("front was observed");
            if seq.phase() == Phase::Paused {
                seq.resume();
            } else if head_dim.is_some() {
                let tokens = self.prompt_ids(seq.id(), seq.prompt_len());
                if let Some(id) = tokens.iter().copied().find(|id| *id >= VOCAB_SIZE) {
                    self.waiting.push_front(seq);
                    return Err(AllocError::TokenOutOfVocab {
                        id,
                        vocab: VOCAB_SIZE,
                    });
                }
                if let Err(err) = seq.allocate_prompt(&mut self.pool) {
                    self.waiting.push_front(seq);
                    return Err(err);
                }
                if let Err(err) = write_model_kv(
                    &mut self.pool,
                    &self.kv.as_ref().expect("storage head was checked").model,
                    &seq,
                    0,
                    &tokens,
                ) {
                    let _ = seq.discard_progress(&mut self.pool);
                    self.waiting.push_front(seq);
                    return Err(err);
                }
                let forward = match forward_tokens(
                    &self.pool,
                    &self.kv.as_ref().expect("storage head was checked").model,
                    &seq,
                    &tokens,
                ) {
                    Ok(forward) => forward,
                    Err(err) => {
                        let _ = seq.discard_progress(&mut self.pool);
                        self.waiting.push_front(seq);
                        return Err(err);
                    }
                };
                let id = seq.id();
                let kv = self.kv.as_mut().expect("storage head was checked");
                kv.pending.insert(id, forward.next_token);
                kv.written.insert(id, tokens);
                kv.scores.insert(id, forward.scores);
                kv.context.insert(id, forward.context);
            } else if let Err(err) = seq.allocate_prompt(&mut self.pool) {
                self.waiting.push_front(seq);
                return Err(err);
            }
            admitted.push(seq.id());
            self.running.push(seq);
        }
        Ok(admitted)
    }

    /// Take the newest decoding sequence out of the batch.
    ///
    /// The victim is placed at the back of the waiting queue, so a request
    /// that was stuck at the head can be admitted next. The oldest running
    /// sequence is left where it is. Finished sequences are not victims.
    /// An empty decode batch returns [`PreemptError::NothingRunning`].
    pub fn preempt(&mut self, policy: PreemptPolicy) -> Result<SequenceId, PreemptError> {
        let index = self
            .running
            .iter()
            .rposition(|seq| seq.phase() == Phase::Decoding)
            .ok_or(PreemptError::NothingRunning)?;
        let mut seq = self.running.remove(index);
        let id = seq.id();
        if policy == PreemptPolicy::Recompute {
            if let Err(err) = seq.discard_progress(&mut self.pool) {
                self.waiting.push_back(seq);
                return Err(PreemptError::Alloc(err));
            }
            if let Some(kv) = self.kv.as_mut() {
                kv.written.remove(&id);
                kv.scores.remove(&id);
                kv.context.remove(&id);
                kv.pending.remove(&id);
            }
        } else {
            seq.pause();
        }
        self.waiting.push_back(seq);
        Ok(id)
    }

    /// Append one generated token to every decoding sequence.
    ///
    /// If the pool cannot hold every new block the batch needs, no sequence
    /// advances. Finished sequences are skipped and stay in the batch. On a
    /// storage pool, the appended id is the one the fixed block chose, and
    /// its K and V are that block's projection. [`EOS`](crate::EOS) finishes
    /// the sequence early.
    pub fn step(&mut self) -> Result<DecodeStep, AllocError> {
        let head_dim = self.storage_head()?;
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
            let mut done = seq.push_token(&mut self.pool)?;
            if head_dim.is_some() {
                let next_id = self
                    .kv
                    .as_ref()
                    .expect("storage head was checked")
                    .pending
                    .get(&id)
                    .copied()
                    .expect("admission ran the block");
                write_model_kv(
                    &mut self.pool,
                    &self.kv.as_ref().expect("storage head was checked").model,
                    seq,
                    seq.len() - 1,
                    &[next_id],
                )?;
                let mut tokens = self
                    .kv
                    .as_ref()
                    .expect("storage head was checked")
                    .written
                    .get(&id)
                    .expect("admission recorded the prompt ids")
                    .clone();
                tokens.push(next_id);
                let forward = forward_tokens(
                    &self.pool,
                    &self.kv.as_ref().expect("storage head was checked").model,
                    seq,
                    &tokens,
                )?;
                let kv = self.kv.as_mut().expect("storage head was checked");
                kv.written
                    .get_mut(&id)
                    .expect("admission recorded the prompt ids")
                    .push(next_id);
                kv.pending.insert(id, forward.next_token);
                kv.scores.insert(id, forward.scores);
                kv.context.insert(id, forward.context);
                if next_id == EOS {
                    seq.finish();
                    done = true;
                }
            }
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
            let id = seq.id();
            retired.push(id);
            if let Some(kv) = self.kv.as_mut() {
                kv.prompts.remove(&id);
                kv.written.remove(&id);
                kv.scores.remove(&id);
                kv.context.remove(&id);
                kv.pending.remove(&id);
            }
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

    /// Sequence at the head of the waiting queue.
    pub fn next_waiting(&self) -> Option<&Sequence> {
        self.waiting.front()
    }

    /// Sequences in the batch, in admission order.
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

fn write_model_kv(
    pool: &mut BlockPool,
    model: &TinyModel,
    seq: &Sequence,
    start: u32,
    tokens: &[u32],
) -> Result<(), AllocError> {
    let block_size = pool.block_size();
    for (offset, token) in tokens.iter().copied().enumerate() {
        let loc = seq.token_loc(block_size, start + offset as u32)?;
        let row = model.kv(token)?;
        pool.kv_mut(loc.block, loc.slot)?.copy_from_slice(&row);
    }
    Ok(())
}

fn forward_tokens(
    pool: &BlockPool,
    model: &TinyModel,
    seq: &Sequence,
    tokens: &[u32],
) -> Result<crate::ModelForward, AllocError> {
    let block_size = pool.block_size();
    let mut rows = Vec::with_capacity(tokens.len());
    for index in 0..tokens.len() as u32 {
        let loc = seq.token_loc(block_size, index)?;
        rows.push(pool.kv(loc.block, loc.slot)?.to_vec());
    }
    model.forward(tokens, &rows)
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

    #[test]
    fn pause_frees_a_batch_slot_and_keeps_generated_tokens() {
        let mut sched = scheduler(8, 4, 1);
        let first = sched.add(4, 3);
        let second = sched.add(4, 1);
        sched.admit().unwrap();
        sched.step().unwrap();
        let blocks = sched.running()[0].blocks().to_vec();
        assert_eq!(sched.running()[0].generated(), 1);
        assert!(sched.admit().unwrap().is_empty());

        let free_before = sched.pool().free_count();
        assert_eq!(sched.preempt(PreemptPolicy::Pause).unwrap(), first);
        assert!(sched.running().is_empty());
        assert_eq!(sched.pool().free_count(), free_before);
        assert_eq!(sched.next_waiting().unwrap().id(), second);

        assert_eq!(sched.admit().unwrap(), vec![second]);
        let parked = sched.next_waiting().unwrap();
        assert_eq!(parked.id(), first);
        assert_eq!(parked.phase(), Phase::Paused);
        assert_eq!(parked.generated(), 1);
        assert_eq!(parked.blocks(), blocks.as_slice());

        assert_eq!(sched.step().unwrap().finished, vec![second]);
        sched.retire().unwrap();
        assert_eq!(sched.admit().unwrap(), vec![first]);
        assert_eq!(sched.running()[0].phase(), Phase::Decoding);
        assert_eq!(sched.running()[0].generated(), 1);
        assert_eq!(sched.running()[0].blocks(), blocks.as_slice());

        assert!(sched.step().unwrap().finished.is_empty());
        assert_eq!(sched.step().unwrap().finished, vec![first]);
        sched.retire().unwrap();
        assert!(sched.is_idle());
        assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    }

    #[test]
    fn recompute_frees_blocks_so_a_waiting_prompt_can_fit() {
        let mut sched = scheduler(1, 8, 2);
        let first = sched.add(4, 2);
        let second = sched.add(8, 0);
        assert_eq!(sched.admit().unwrap(), vec![first]);
        sched.step().unwrap();
        assert_eq!(sched.running()[0].generated(), 1);
        assert!(sched.admit().unwrap().is_empty());
        assert_eq!(sched.pool().free_count(), 0);

        assert_eq!(sched.preempt(PreemptPolicy::Recompute).unwrap(), first);
        assert!(sched.running().is_empty());
        assert_eq!(sched.pool().free_count(), 1);
        assert_eq!(sched.next_waiting().unwrap().id(), second);

        assert_eq!(sched.admit().unwrap(), vec![second]);
        assert_eq!(sched.running()[0].phase(), Phase::Finished);
        assert_eq!(sched.running()[0].len(), 8);
        let parked = sched.next_waiting().unwrap();
        assert_eq!(parked.id(), first);
        assert_eq!(parked.phase(), Phase::Waiting);
        assert_eq!(parked.generated(), 0);
        assert_eq!(parked.len(), 0);

        sched.retire().unwrap();
        assert_eq!(sched.admit().unwrap(), vec![first]);
        assert_eq!(sched.running()[0].generated(), 0);
        assert_eq!(sched.running()[0].len(), 4);
        assert!(sched.step().unwrap().finished.is_empty());
        assert_eq!(sched.step().unwrap().finished, vec![first]);
        sched.retire().unwrap();
        assert!(sched.is_idle());
        assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    }

    #[test]
    fn the_oldest_running_sequence_stays_when_a_newer_one_is_preempted() {
        let mut sched = scheduler(8, 4, 2);
        let first = sched.add(4, 2);
        let second = sched.add(4, 2);
        let third = sched.add(4, 1);
        sched.admit().unwrap();
        sched.step().unwrap();

        assert_eq!(sched.preempt(PreemptPolicy::Pause).unwrap(), second);
        assert_eq!(sched.running().len(), 1);
        assert_eq!(sched.running()[0].id(), first);
        assert_eq!(sched.running()[0].generated(), 1);

        assert_eq!(sched.admit().unwrap(), vec![third]);
        assert_eq!(sched.next_waiting().unwrap().id(), second);
        assert_eq!(sched.next_waiting().unwrap().phase(), Phase::Paused);
        assert_eq!(sched.next_waiting().unwrap().generated(), 1);
    }

    #[test]
    fn preempt_with_nothing_decoding_fails() {
        let mut sched = scheduler(4, 4, 2);
        assert_eq!(
            sched.preempt(PreemptPolicy::Pause).unwrap_err(),
            PreemptError::NothingRunning
        );

        let finished = sched.add(4, 0);
        sched.admit().unwrap();
        assert_eq!(sched.running()[0].phase(), Phase::Finished);
        assert_eq!(
            sched.preempt(PreemptPolicy::Recompute).unwrap_err(),
            PreemptError::NothingRunning
        );
        assert_eq!(sched.running()[0].id(), finished);
        assert_eq!(sched.waiting_len(), 0);
        assert_eq!(sched.retire().unwrap(), vec![finished]);
        assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    }

    #[test]
    fn a_generation_loop_writes_nonzero_kv() {
        let head_dim = 2;
        let model = TinyModel::new(head_dim);
        let mut sched = Scheduler::new(BlockPool::with_storage(8, 4, head_dim * 2), 1);
        let prompt = [1u32, 2, 3];
        let id = sched.add_tokens(&prompt, 2);

        assert_eq!(sched.admit().unwrap(), vec![id]);
        assert_eq!(sched.token_ids(id).unwrap(), &prompt);
        assert_eq!(sched.scores(id).unwrap().len(), prompt.len());
        assert_stored_matches_model(&sched, &model);

        let mut expected = prompt.to_vec();
        let chosen = model
            .forward(&expected, &rows_for(&model, &expected))
            .unwrap()
            .next_token;
        let step = sched.step().unwrap();
        assert!(step.finished.is_empty());
        expected.push(chosen);
        assert_eq!(sched.token_ids(id).unwrap(), expected.as_slice());
        assert_eq!(sched.scores(id).unwrap().len(), expected.len());
        assert_stored_matches_model(&sched, &model);

        let chosen = model
            .forward(&expected, &rows_for(&model, &expected))
            .unwrap()
            .next_token;
        let step = sched.step().unwrap();
        assert_eq!(step.finished, vec![id]);
        expected.push(chosen);
        assert_eq!(sched.token_ids(id).unwrap(), expected.as_slice());
        assert_ne!(&expected[prompt.len()..], &[4, 5]);
        assert_stored_matches_model(&sched, &model);

        assert_eq!(sched.retire().unwrap(), vec![id]);
        assert!(sched.is_idle());
        assert!(sched.scores(id).is_none());
        assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    }

    fn rows_for(model: &TinyModel, tokens: &[u32]) -> Vec<Vec<f32>> {
        tokens
            .iter()
            .copied()
            .map(|token| model.kv(token).unwrap())
            .collect()
    }

    fn assert_stored_matches_model(sched: &Scheduler, model: &TinyModel) {
        let block_size = sched.pool().block_size();
        let seq = &sched.running()[0];
        let ids = sched.token_ids(seq.id()).unwrap();
        for (index, token) in ids.iter().copied().enumerate() {
            let loc = seq.token_loc(block_size, index as u32).unwrap();
            let stored = sched.pool().kv(loc.block, loc.slot).unwrap();
            assert_eq!(stored, model.kv(token).unwrap().as_slice());
            assert!(stored.iter().any(|value| *value != 0.0));
        }
    }

    #[test]
    fn a_storage_width_that_is_not_two_halves_admits_nothing() {
        let mut sched = Scheduler::new(BlockPool::with_storage(2, 4, 3), 1);
        let id = sched.add_tokens(&[1], 1);
        assert_eq!(
            sched.admit().unwrap_err(),
            AllocError::KvWidth {
                floats_per_token: 3,
                head_dim: 1,
            }
        );
        assert!(sched.running().is_empty());
        assert_eq!(sched.next_waiting().unwrap().id(), id);
        assert_eq!(sched.pool().free_count(), sched.pool().capacity());
        assert_eq!(
            sched.step().unwrap_err(),
            AllocError::KvWidth {
                floats_per_token: 3,
                head_dim: 1,
            }
        );
    }

    #[test]
    fn a_token_outside_the_vocabulary_admits_nothing() {
        let mut sched = Scheduler::new(BlockPool::with_storage(4, 4, 4), 1);
        let id = sched.add_tokens(&[VOCAB_SIZE], 0);
        assert_eq!(
            sched.admit().unwrap_err(),
            AllocError::TokenOutOfVocab {
                id: VOCAB_SIZE,
                vocab: VOCAB_SIZE,
            }
        );
        assert!(sched.running().is_empty());
        assert_eq!(sched.next_waiting().unwrap().id(), id);
        assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    }
}
