//! Paged KV-cache block management.
//!
//! The pool tracks fixed-size blocks for a key/value cache.
//! Callers own the KV bytes and address them by block id.

/// Identifier of one block in a [`BlockPool`].
///
/// Ids are local to the pool that issued them. Copying an id does not
/// change ownership of the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(u32);

impl BlockId {
    /// Index of this block in the pool, in `0..capacity`.
    pub fn index(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Failure from a block-pool operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    /// Every block is currently allocated.
    Exhausted,
    /// The id is outside this pool.
    InvalidBlock(BlockId),
    /// `free` was called on a block that is not allocated.
    NotAllocated(BlockId),
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocError::Exhausted => write!(f, "block pool exhausted"),
            AllocError::InvalidBlock(id) => write!(f, "block {id} is outside the pool"),
            AllocError::NotAllocated(id) => write!(f, "block {id} is not allocated"),
        }
    }
}

impl std::error::Error for AllocError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Allocated,
}

/// Pool of fixed-size KV-cache blocks.
///
/// Each block stands for `block_size` token slots. This type owns allocation
/// metadata only. KV tensors live with the caller, keyed by [`BlockId`].
#[derive(Debug)]
pub struct BlockPool {
    block_size: u32,
    slots: Vec<SlotState>,
}

impl BlockPool {
    /// Create a pool of `num_blocks` blocks, each `block_size` tokens wide.
    ///
    /// # Panics
    /// Panics if `block_size` is zero.
    pub fn new(num_blocks: u32, block_size: u32) -> Self {
        assert!(block_size > 0, "block_size must be non-zero");
        Self {
            block_size,
            slots: vec![SlotState::Free; num_blocks as usize],
        }
    }

    /// Tokens represented by one block.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Total number of blocks in the pool, free and allocated.
    pub fn capacity(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Take one free block.
    ///
    /// The caller uniquely owns the returned id until [`BlockPool::free`].
    pub fn allocate(&mut self) -> Result<BlockId, AllocError> {
        let index = self
            .slots
            .iter()
            .position(|slot| *slot == SlotState::Free)
            .ok_or(AllocError::Exhausted)?;
        self.slots[index] = SlotState::Allocated;
        Ok(BlockId(index as u32))
    }

    /// Return a block to the pool.
    ///
    /// The id must refer to a block this pool previously allocated and has
    /// not yet freed.
    pub fn free(&mut self, id: BlockId) -> Result<(), AllocError> {
        let slot = self
            .slots
            .get_mut(id.0 as usize)
            .ok_or(AllocError::InvalidBlock(id))?;
        match *slot {
            SlotState::Free => Err(AllocError::NotAllocated(id)),
            SlotState::Allocated => {
                *slot = SlotState::Free;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_free_round_trip() {
        let mut pool = BlockPool::new(4, 16);
        let id = pool.allocate().expect("pool has free blocks");
        assert_eq!(id.index(), 0);
        pool.free(id).expect("block is live");
        let again = pool.allocate().expect("freed block is reusable");
        assert_eq!(again, id);
    }

    #[test]
    fn two_live_blocks_have_distinct_ids() {
        let mut pool = BlockPool::new(2, 8);
        let first = pool.allocate().expect("first block");
        let second = pool.allocate().expect("second block");
        assert_ne!(first, second);
        pool.free(first).expect("first block is live");
        pool.free(second).expect("second block is live");
    }
}
