//! One generation request and the blocks that hold its KV cache.
//!
//! A sequence splits a token length into the prompt that was submitted and
//! the tokens generated after admission. The block table underneath is still
//! a single run of blocks.

use crate::{AllocError, BlockPool, BlockTable, TokenLoc};

/// Identifier assigned in arrival order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(u64);

impl SequenceId {
    /// The raw id.
    pub fn get(self) -> u64 {
        self.0
    }

    /// Build an id. The scheduler assigns these in arrival order.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for SequenceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Where a sequence is in its lifetime.
///
/// There is no separate prefill phase. Admission reserves the whole prompt
/// at once, then decode appends one token at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Queued. No blocks are reserved.
    Waiting,
    /// Prompt blocks are reserved and decode tokens are still owed.
    Decoding,
    /// The token budget is spent. Blocks stay reserved until the owner releases them.
    Finished,
    /// Out of the batch, with its blocks and generated tokens kept.
    ///
    /// [`crate::Scheduler::preempt`] with [`crate::PreemptPolicy::Pause`]
    /// enters this phase. The next admission resumes decode without
    /// reserving the prompt again.
    Paused,
}

/// Prompt tokens plus generated tokens for one request.
///
/// The table is empty while the sequence is [`Phase::Waiting`]. After
/// admission, [`Sequence::len`] matches `prompt_len + generated`.
/// [`Sequence::release`] returns the blocks to the pool. Dropping a sequence
/// that holds blocks without releasing leaks those references for as long as
/// the pool lives.
#[derive(Debug)]
pub struct Sequence {
    id: SequenceId,
    prompt_len: u32,
    generated: u32,
    max_new_tokens: u32,
    phase: Phase,
    table: BlockTable,
}

impl Sequence {
    /// A waiting sequence. It holds no blocks yet.
    pub fn new(id: SequenceId, prompt_len: u32, max_new_tokens: u32) -> Self {
        Self {
            id,
            prompt_len,
            generated: 0,
            max_new_tokens,
            phase: Phase::Waiting,
            table: BlockTable::new(),
        }
    }

    /// Arrival id.
    pub fn id(&self) -> SequenceId {
        self.id
    }

    /// Tokens submitted with the request.
    pub fn prompt_len(&self) -> u32 {
        self.prompt_len
    }

    /// Decode tokens appended so far.
    pub fn generated(&self) -> u32 {
        self.generated
    }

    /// Decode tokens this request is allowed to produce.
    pub fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens
    }

    /// Current lifetime phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Tokens currently reserved in the block table.
    ///
    /// This is `0` while the sequence is waiting. After admission it is
    /// `prompt_len + generated`.
    pub fn len(&self) -> u32 {
        self.table.len()
    }

    /// `true` when no blocks are reserved.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Blocks reserved for this sequence, in token order.
    pub fn blocks(&self) -> &[crate::BlockId] {
        self.table.blocks()
    }

    /// Reserve blocks for the prompt.
    ///
    /// A request with `max_new_tokens == 0` finishes immediately. The table
    /// is unchanged when allocation fails. This does not write KV vectors.
    pub fn allocate_prompt(&mut self, pool: &mut BlockPool) -> Result<(), AllocError> {
        debug_assert_eq!(self.phase, Phase::Waiting);
        debug_assert_eq!(self.generated, 0);
        self.table.grow(pool, self.prompt_len)?;
        self.enter_after_prompt();
        Ok(())
    }

    /// Reserve the prompt with [`crate::fill_tokens`] and write each id.
    ///
    /// `token_ids.len()` is the prompt length. On failure the table is
    /// released and the sequence stays waiting.
    pub(crate) fn fill_prompt(
        &mut self,
        pool: &mut BlockPool,
        head_dim: u32,
        token_ids: &[u32],
    ) -> Result<(), AllocError> {
        debug_assert_eq!(self.phase, Phase::Waiting);
        debug_assert_eq!(self.generated, 0);
        debug_assert_eq!(token_ids.len() as u32, self.prompt_len);
        if let Err(err) = crate::fill_tokens(pool, &mut self.table, head_dim, token_ids) {
            let table = std::mem::take(&mut self.table);
            let _ = table.release(pool);
            return Err(err);
        }
        self.enter_after_prompt();
        Ok(())
    }

    /// Where logical token `index` lives.
    ///
    /// `block_size` must be the block size of the pool this sequence grew on.
    pub fn token_loc(&self, block_size: u32, index: u32) -> Result<TokenLoc, AllocError> {
        if index >= self.table.len() {
            return Err(AllocError::TokenOutOfRange {
                index,
                len: self.table.len(),
            });
        }
        let block_index = (index / block_size) as usize;
        let Some(block) = self.table.blocks().get(block_index).copied() else {
            return Err(AllocError::TokenOutOfRange {
                index,
                len: self.table.len(),
            });
        };
        Ok(TokenLoc {
            block,
            slot: index % block_size,
        })
    }

    pub(crate) fn table(&self) -> &BlockTable {
        &self.table
    }

    fn enter_after_prompt(&mut self) {
        self.phase = if self.max_new_tokens == 0 {
            Phase::Finished
        } else {
            Phase::Decoding
        };
    }

    /// Append one generated token.
    ///
    /// Returns whether this token spent the budget. The table is unchanged
    /// when allocation fails.
    pub fn push_token(&mut self, pool: &mut BlockPool) -> Result<bool, AllocError> {
        debug_assert_eq!(self.phase, Phase::Decoding);
        debug_assert_eq!(self.table.len(), self.prompt_len + self.generated);
        let new_len = self
            .table
            .len()
            .checked_add(1)
            .expect("sequence length fits in u32");
        self.table.grow(pool, new_len)?;
        self.generated += 1;
        let finished = self.generated == self.max_new_tokens;
        if finished {
            self.phase = Phase::Finished;
        }
        Ok(finished)
    }

    /// Drop this sequence's block references.
    pub fn release(self, pool: &mut BlockPool) -> Result<(), AllocError> {
        self.table.release(pool)
    }

    pub(crate) fn pause(&mut self) {
        debug_assert_eq!(self.phase, Phase::Decoding);
        self.phase = Phase::Paused;
    }

    pub(crate) fn resume(&mut self) {
        debug_assert_eq!(self.phase, Phase::Paused);
        self.phase = Phase::Decoding;
    }

    /// Free this sequence's blocks and forget generated tokens.
    ///
    /// The sequence is [`Phase::Waiting`] afterwards. If releasing a block
    /// fails, the table is still gone and the phase is still waiting.
    pub(crate) fn discard_progress(&mut self, pool: &mut BlockPool) -> Result<(), AllocError> {
        debug_assert!(matches!(
            self.phase,
            Phase::Decoding | Phase::Paused | Phase::Finished
        ));
        let table = std::mem::take(&mut self.table);
        let released = table.release(pool);
        self.generated = 0;
        self.phase = Phase::Waiting;
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_and_generated_tokens_share_one_table() {
        let mut pool = BlockPool::new(4, 4);
        let mut seq = Sequence::new(SequenceId::new(7), 6, 3);
        assert_eq!(seq.phase(), Phase::Waiting);
        assert_eq!(seq.len(), 0);
        assert!(seq.blocks().is_empty());

        seq.allocate_prompt(&mut pool).unwrap();
        assert_eq!(seq.phase(), Phase::Decoding);
        assert_eq!(seq.len(), 6);
        assert_eq!(seq.generated(), 0);
        assert_eq!(seq.blocks().len(), 2);

        assert!(!seq.push_token(&mut pool).unwrap());
        assert_eq!(seq.generated(), 1);
        assert_eq!(seq.len(), 7);
        assert_eq!(seq.phase(), Phase::Decoding);

        assert!(!seq.push_token(&mut pool).unwrap());
        assert!(seq.push_token(&mut pool).unwrap());
        assert_eq!(seq.generated(), 3);
        assert_eq!(seq.len(), 9);
        assert_eq!(seq.phase(), Phase::Finished);
        assert_eq!(seq.prompt_len(), 6);

        seq.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn zero_new_tokens_finishes_when_the_prompt_is_reserved() {
        let mut pool = BlockPool::new(2, 8);
        let mut seq = Sequence::new(SequenceId::new(1), 8, 0);
        seq.allocate_prompt(&mut pool).unwrap();
        assert_eq!(seq.phase(), Phase::Finished);
        assert_eq!(seq.generated(), 0);
        assert_eq!(seq.len(), 8);
        seq.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn a_failed_prompt_reservation_leaves_the_sequence_waiting() {
        let mut pool = BlockPool::new(1, 4);
        let mut seq = Sequence::new(SequenceId::new(1), 8, 2);
        assert_eq!(
            seq.allocate_prompt(&mut pool).unwrap_err(),
            AllocError::Exhausted
        );
        assert_eq!(seq.phase(), Phase::Waiting);
        assert_eq!(seq.len(), 0);
        assert_eq!(pool.free_count(), pool.capacity());
    }
}
