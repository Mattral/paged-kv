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
    /// The block is not currently allocated.
    NotAllocated(BlockId),
    /// `used` is greater than the block width.
    InvalidUsed {
        /// Block the caller tried to update.
        block: BlockId,
        /// Requested initialized-token count.
        used: u32,
        /// Width of every block in the pool.
        block_size: u32,
    },
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocError::Exhausted => write!(f, "block pool exhausted"),
            AllocError::InvalidBlock(id) => write!(f, "block {id} is outside the pool"),
            AllocError::NotAllocated(id) => write!(f, "block {id} is not allocated"),
            AllocError::InvalidUsed {
                block,
                used,
                block_size,
            } => write!(
                f,
                "block {block} cannot hold {used} tokens (block size is {block_size})"
            ),
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
///
/// Blocks are interchangeable, so a hole on the free list is not external
/// fragmentation. The only waste is internal: allocated blocks whose
/// initialized token count is below `block_size`. [`BlockPool::wasted_slots`]
/// reports that tail.
#[derive(Debug)]
pub struct BlockPool {
    block_size: u32,
    /// `true` while the block is checked out.
    allocated: Vec<bool>,
    /// Initialized token slots in each block. Zero when the block is free.
    used: Vec<u32>,
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
            used: vec![0; num_blocks as usize],
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
        debug_assert_eq!(self.used[index as usize], 0, "free block still has tokens");
        self.allocated[index as usize] = true;
        Ok(BlockId(index))
    }

    /// Record how many token slots in `id` hold live KV data.
    ///
    /// `used` must be at most [`BlockPool::block_size`]. A freshly allocated
    /// block starts at zero, which counts as a fully wasted block until the
    /// caller fills it. Every owner of a block sees the same count.
    pub fn set_used(&mut self, id: BlockId, used: u32) -> Result<(), AllocError> {
        self.ensure_live(id)?;
        if used > self.block_size {
            return Err(AllocError::InvalidUsed {
                block: id,
                used,
                block_size: self.block_size,
            });
        }
        self.used[id.0 as usize] = used;
        Ok(())
    }

    /// Initialized token slots in an allocated block.
    pub fn used(&self, id: BlockId) -> Result<u32, AllocError> {
        self.ensure_live(id)?;
        Ok(self.used[id.0 as usize])
    }

    /// Token slots in allocated blocks that do not hold live KV data.
    ///
    /// Free blocks contribute nothing.
    pub fn wasted_slots(&self) -> u64 {
        self.allocated
            .iter()
            .zip(self.used.iter())
            .filter(|(live, _)| **live)
            .map(|(_, used)| u64::from(self.block_size - *used))
            .sum()
    }

    /// `wasted_slots / (allocated blocks * block_size)`.
    ///
    /// This is internal fragmentation only. An empty pool reports `0.0`.
    pub fn fragmentation(&self) -> f64 {
        let allocated = u64::from(self.capacity() - self.free_count());
        if allocated == 0 {
            0.0
        } else {
            self.wasted_slots() as f64 / (allocated as f64 * f64::from(self.block_size))
        }
    }

    fn ensure_live(&self, id: BlockId) -> Result<(), AllocError> {
        match self.allocated.get(id.0 as usize) {
            None => Err(AllocError::InvalidBlock(id)),
            Some(false) => Err(AllocError::NotAllocated(id)),
            Some(true) => Ok(()),
        }
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
        self.used[id.0 as usize] = 0;
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

    #[test]
    fn partial_block_reports_tail_waste() {
        let mut pool = BlockPool::new(2, 16);
        let id = pool.allocate().unwrap();
        assert_eq!(pool.used(id).unwrap(), 0);
        assert_eq!(pool.wasted_slots(), 16);

        pool.set_used(id, 10).unwrap();
        assert_eq!(pool.used(id).unwrap(), 10);
        assert_eq!(pool.wasted_slots(), 6);
        assert_eq!(pool.fragmentation(), 6.0 / 16.0);
    }

    #[test]
    fn full_block_has_no_internal_fragmentation() {
        let mut pool = BlockPool::new(1, 8);
        let id = pool.allocate().unwrap();
        pool.set_used(id, 8).unwrap();
        assert_eq!(pool.wasted_slots(), 0);
        assert_eq!(pool.fragmentation(), 0.0);
    }

    #[test]
    fn freeing_a_partial_block_drops_its_waste() {
        let mut pool = BlockPool::new(2, 8);
        let first = pool.allocate().unwrap();
        let second = pool.allocate().unwrap();
        pool.set_used(first, 3).unwrap();
        pool.set_used(second, 8).unwrap();
        assert_eq!(pool.wasted_slots(), 5);

        pool.free(first).unwrap();
        assert_eq!(pool.wasted_slots(), 0);
        assert_eq!(pool.fragmentation(), 0.0);

        let reused = pool.allocate().unwrap();
        assert_eq!(reused, first);
        assert_eq!(pool.used(reused).unwrap(), 0);
        assert_eq!(pool.wasted_slots(), 8);
    }

    #[test]
    fn set_used_rejects_free_oversized_and_unknown_blocks() {
        let mut pool = BlockPool::new(1, 4);
        let id = pool.allocate().unwrap();
        pool.free(id).unwrap();
        assert_eq!(
            pool.set_used(id, 1).unwrap_err(),
            AllocError::NotAllocated(id)
        );
        assert_eq!(pool.used(id).unwrap_err(), AllocError::NotAllocated(id));

        let live = pool.allocate().unwrap();
        assert_eq!(
            pool.set_used(live, 5).unwrap_err(),
            AllocError::InvalidUsed {
                block: live,
                used: 5,
                block_size: 4,
            }
        );
        assert_eq!(pool.used(live).unwrap(), 0);

        let bogus = BlockId(3);
        assert_eq!(
            pool.used(bogus).unwrap_err(),
            AllocError::InvalidBlock(bogus)
        );
    }

    #[test]
    fn idle_pool_has_zero_fragmentation() {
        let pool = BlockPool::new(4, 8);
        assert_eq!(pool.wasted_slots(), 0);
        assert_eq!(pool.fragmentation(), 0.0);
    }
}
