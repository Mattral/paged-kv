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

/// Pool of fixed-size KV-cache blocks.
///
/// Each block stands for `block_size` token slots. This type owns allocation
/// metadata only. KV tensors live with the caller, keyed by [`BlockId`].
///
/// Free ids live on an explicit stack. [`BlockPool::allocate`] pops that
/// stack, so the block just released is the next one handed out. A fresh
/// pool is stacked from high ids to low ids, and the first allocations are
/// `0, 1, 2, ...`.
#[derive(Debug)]
pub struct BlockPool {
    block_size: u32,
    /// `true` while the block is checked out.
    allocated: Vec<bool>,
    /// Free block indices. The next allocation pops the last entry.
    free_list: Vec<u32>,
}

impl BlockPool {
    /// Create a pool of `num_blocks` blocks, each `block_size` tokens wide.
    ///
    /// # Panics
    /// Panics if `block_size` is zero.
    pub fn new(num_blocks: u32, block_size: u32) -> Self {
        assert!(block_size > 0, "block_size must be non-zero");
        let mut free_list = Vec::with_capacity(num_blocks as usize);
        for id in (0..num_blocks).rev() {
            free_list.push(id);
        }
        Self {
            block_size,
            allocated: vec![false; num_blocks as usize],
            free_list,
        }
    }

    /// Tokens represented by one block.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Total number of blocks in the pool, free and allocated.
    pub fn capacity(&self) -> u32 {
        self.allocated.len() as u32
    }

    /// Blocks that [`BlockPool::allocate`] can still hand out.
    pub fn free_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Take one free block.
    ///
    /// The caller uniquely owns the returned id until [`BlockPool::free`].
    pub fn allocate(&mut self) -> Result<BlockId, AllocError> {
        let index = self.free_list.pop().ok_or(AllocError::Exhausted)?;
        debug_assert!(
            !self.allocated[index as usize],
            "free list contained a live block"
        );
        self.allocated[index as usize] = true;
        Ok(BlockId(index))
    }

    /// Return a block to the pool.
    ///
    /// The id must refer to a block this pool previously allocated and has
    /// not yet freed. The freed id is pushed onto the free list.
    pub fn free(&mut self, id: BlockId) -> Result<(), AllocError> {
        let slot = self
            .allocated
            .get_mut(id.0 as usize)
            .ok_or(AllocError::InvalidBlock(id))?;
        if !*slot {
            return Err(AllocError::NotAllocated(id));
        }
        *slot = false;
        self.free_list.push(id.0);
        Ok(())
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
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn most_recently_freed_block_is_reused_first() {
        let mut pool = BlockPool::new(4, 8);
        let first = pool.allocate().unwrap();
        let second = pool.allocate().unwrap();
        let third = pool.allocate().unwrap();
        assert_eq!((first.index(), second.index(), third.index()), (0, 1, 2));
        assert_eq!(pool.free_count(), 1);

        pool.free(second).unwrap();
        assert_eq!(pool.free_count(), 2);
        assert_eq!(pool.allocate().unwrap(), second);
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn freeing_the_last_live_block_clears_exhaustion() {
        let mut pool = BlockPool::new(1, 2);
        let id = pool.allocate().unwrap();
        assert_eq!(pool.allocate().unwrap_err(), AllocError::Exhausted);
        assert_eq!(pool.free_count(), 0);

        pool.free(id).unwrap();
        assert_eq!(pool.free_count(), 1);
        assert_eq!(pool.allocate().unwrap(), id);
    }

    #[test]
    fn double_free_is_rejected_and_does_not_grow_the_free_list() {
        let mut pool = BlockPool::new(1, 2);
        let id = pool.allocate().unwrap();
        pool.free(id).unwrap();
        assert_eq!(pool.free(id).unwrap_err(), AllocError::NotAllocated(id));
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn free_rejects_an_id_outside_the_pool() {
        let mut pool = BlockPool::new(2, 2);
        let bogus = BlockId(9);
        assert_eq!(
            pool.free(bogus).unwrap_err(),
            AllocError::InvalidBlock(bogus)
        );
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn empty_pool_allocates_nothing() {
        let mut pool = BlockPool::new(0, 4);
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.allocate().unwrap_err(), AllocError::Exhausted);
    }
}
