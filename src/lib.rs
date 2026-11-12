//! Paged KV-cache block management.
//!
//! [`BlockPool`] tracks fixed-size blocks. [`BlockTable`] maps one sequence's
//! tokens onto those blocks and can fork a shared prefix. [`PrefixCache`]
//! stores those forks under a caller-supplied key and counts hits.
//! [`Scheduler`] admits sequences first-come, first-served, retires them
//! when their token budget is spent, and can preempt the newest decoding
//! sequence when the caller asks. [`BlockPool::new`] tracks ids only.
//! [`BlockPool::with_storage`] also keeps one `f32` vector per token slot,
//! addressed by [`BlockId`]. On that pool, admission and each decode step
//! write K and V from [`TinyModel`] and take the next token from that block.
//! [`encode`] and [`decode`] map text to bytes and back. [`generate`] runs
//! that block on a string and returns the ids it appended. [`ContiguousTable`]
//! is the non-paged baseline: one length and one capacity, with no blocks.

mod attention;
mod contiguous;
mod prefix_cache;
mod scheduler;
mod sequence;
mod tiny_model;
mod tokenizer;

pub use attention::{attend, fill_tokens, write_token, Attention};
pub use contiguous::ContiguousTable;
pub use prefix_cache::PrefixCache;
pub use scheduler::{generate, DecodeStep, Generated, PreemptError, PreemptPolicy, Scheduler};
pub use sequence::{Phase, Sequence, SequenceId};
pub use tiny_model::{ModelForward, TinyModel};
pub use tokenizer::{decode, encode, EOS, VOCAB_SIZE};

/// Block id and slot of one token inside a [`BlockTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenLoc {
    /// Block that holds the token.
    pub block: BlockId,
    /// Slot inside that block, in `0..block_size`.
    pub slot: u32,
}

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

    /// Build an id from a pool index.
    ///
    /// This does not allocate. The pool checks the index when it is used.
    pub fn from_index(index: u32) -> Self {
        Self(index)
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Failure from the block pool or a block table.
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
    /// The refcount would exceed `u32::MAX`.
    RefcountOverflow(BlockId),
    /// `grow` was asked to shorten a table.
    LengthRegression {
        /// Current logical token count.
        current: u32,
        /// Requested token count.
        requested: u32,
    },
    /// A prefix was longer than the table it came from.
    PrefixOutOfRange {
        /// Requested prefix length, in tokens.
        prefix_len: u32,
        /// Logical token count of the source table.
        len: u32,
    },
    /// [`BlockPool::kv`] was used on a pool built with [`BlockPool::new`].
    NoStorage,
    /// The token slot is outside the block.
    InvalidSlot {
        /// Block the caller addressed.
        block: BlockId,
        /// Slot inside that block.
        slot: u32,
        /// Width of every block in the pool.
        block_size: u32,
    },
    /// A write was aimed at a block another owner still holds.
    ///
    /// Copy the block with [`BlockPool::cow`] and write the new id.
    SharedWrite(BlockId),
    /// A token index is past the table that would hold it.
    TokenOutOfRange {
        /// Requested token index.
        index: u32,
        /// Logical token count of the table.
        len: u32,
    },
    /// The stored vector is not two halves of `head_dim`.
    KvWidth {
        /// Width stored by the pool.
        floats_per_token: u32,
        /// Head width the attention step was given.
        head_dim: u32,
    },
    /// A token id is outside the byte vocabulary.
    TokenOutOfVocab {
        /// The id that was presented.
        id: u32,
        /// [`VOCAB_SIZE`](crate::VOCAB_SIZE).
        vocab: u32,
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
            AllocError::RefcountOverflow(id) => {
                write!(f, "refcount of block {id} overflowed")
            }
            AllocError::LengthRegression { current, requested } => write!(
                f,
                "cannot shrink a block table from {current} tokens to {requested}"
            ),
            AllocError::PrefixOutOfRange { prefix_len, len } => write!(
                f,
                "prefix of {prefix_len} tokens exceeds the table length of {len}"
            ),
            AllocError::NoStorage => write!(f, "this pool holds no kv vectors"),
            AllocError::InvalidSlot {
                block,
                slot,
                block_size,
            } => write!(
                f,
                "block {block} has no slot {slot} (block size is {block_size})"
            ),
            AllocError::SharedWrite(id) => {
                write!(f, "block {id} is shared; copy it before writing kv")
            }
            AllocError::TokenOutOfRange { index, len } => {
                write!(f, "token {index} is outside a table of length {len}")
            }
            AllocError::KvWidth {
                floats_per_token,
                head_dim,
            } => write!(
                f,
                "kv width {floats_per_token} is not two halves of head dim {head_dim}"
            ),
            AllocError::TokenOutOfVocab { id, vocab } => {
                write!(f, "token {id} is outside a vocabulary of {vocab}")
            }
        }
    }
}

impl std::error::Error for AllocError {}

/// Pool of fixed-size KV-cache blocks.
///
/// Each block stands for `block_size` token slots. [`BlockPool::new`] stores
/// only allocation metadata. [`BlockPool::with_storage`] also stores one
/// `f32` vector in every token slot. A model forward pass writes that vector
/// through [`BlockPool::kv_mut`].
///
/// Free ids live on an explicit stack. [`BlockPool::allocate`] pops that
/// stack, so the block just released is the next one handed out. A fresh
/// pool is stacked from high ids to low ids, and the first allocations are
/// `0, 1, 2, ...`.
///
/// Blocks are interchangeable, so a hole on the free list is not external
/// fragmentation. The only waste is internal: allocated blocks whose
/// initialized token count is below `block_size`. [`BlockPool::wasted_slots`]
/// reports that tail once per physical block, not once per owner.
///
/// A block with a refcount above one is shared. Every owner observes the
/// same initialized slots. [`BlockPool::free`] drops a single owner and
/// recycles the block only at zero. [`BlockPool::cow`] forks a private
/// block before a write that would diverge from the other owners.
#[derive(Debug)]
pub struct BlockPool {
    block_size: u32,
    /// Live owners of each block. Zero means the id is on `free_list`.
    refcounts: Vec<u32>,
    /// Initialized token slots in each block. Zero when the block is free.
    used: Vec<u32>,
    /// Free block indices. The next allocation pops the last entry.
    free_list: Vec<u32>,
    /// `f32` KV vectors, present only for [`BlockPool::with_storage`].
    kv: Option<KvSlab>,
}

#[derive(Debug)]
struct KvSlab {
    floats_per_token: u32,
    data: Vec<f32>,
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
            refcounts: vec![0; num_blocks as usize],
            used: vec![0; num_blocks as usize],
            free_list,
            kv: None,
        }
    }

    /// Pool that stores an `f32` vector in every token slot.
    ///
    /// `floats_per_token` is the width of that vector. A block holds
    /// `block_size` of them, packed in slot order. Fresh slots are zero.
    /// Freeing the last owner zeroes the block again.
    ///
    /// # Panics
    /// Panics if `block_size` or `floats_per_token` is zero, or if the slab
    /// does not fit in memory.
    pub fn with_storage(num_blocks: u32, block_size: u32, floats_per_token: u32) -> Self {
        assert!(floats_per_token > 0, "floats_per_token must be non-zero");
        let mut pool = Self::new(num_blocks, block_size);
        let cells = (num_blocks as usize)
            .checked_mul(block_size as usize)
            .and_then(|n| n.checked_mul(floats_per_token as usize))
            .expect("kv slab fits in memory");
        pool.kv = Some(KvSlab {
            floats_per_token,
            data: vec![0.0; cells],
        });
        pool
    }

    /// Width of each stored token vector, if this pool has one.
    pub fn floats_per_token(&self) -> Option<u32> {
        self.kv.as_ref().map(|kv| kv.floats_per_token)
    }

    /// Tokens represented by one block.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Number of blocks required to hold `len` tokens of `block_size`.
    ///
    /// # Panics
    /// Panics if `block_size` is zero.
    pub fn blocks_needed(len: u32, block_size: u32) -> u32 {
        assert!(block_size > 0, "block_size must be non-zero");
        let len = u64::from(len);
        let block_size = u64::from(block_size);
        len.div_ceil(block_size) as u32
    }

    /// Total number of blocks in the pool, free and allocated.
    pub fn capacity(&self) -> u32 {
        self.refcounts.len() as u32
    }

    /// Blocks that [`BlockPool::allocate`] can still hand out.
    pub fn free_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Take one free block and give it a refcount of one.
    ///
    /// Copying the returned [`BlockId`] does not add an owner. Share the
    /// block with [`BlockPool::acquire`].
    pub fn allocate(&mut self) -> Result<BlockId, AllocError> {
        let index = self.free_list.pop().ok_or(AllocError::Exhausted)?;
        debug_assert_eq!(
            self.refcounts[index as usize], 0,
            "free list contained a live block"
        );
        debug_assert_eq!(self.used[index as usize], 0, "free block still has tokens");
        self.refcounts[index as usize] = 1;
        Ok(BlockId(index))
    }

    /// Add one owner to an allocated block.
    ///
    /// The new owner shares the block's current contents. Returns the new
    /// refcount.
    pub fn acquire(&mut self, id: BlockId) -> Result<u32, AllocError> {
        let rc = self
            .refcounts
            .get_mut(id.0 as usize)
            .ok_or(AllocError::InvalidBlock(id))?;
        if *rc == 0 {
            return Err(AllocError::NotAllocated(id));
        }
        let next = rc.checked_add(1).ok_or(AllocError::RefcountOverflow(id))?;
        *rc = next;
        Ok(next)
    }

    /// Owners of an in-range block. A free block reports zero.
    pub fn refcount(&self, id: BlockId) -> Result<u32, AllocError> {
        self.refcounts
            .get(id.0 as usize)
            .copied()
            .ok_or(AllocError::InvalidBlock(id))
    }

    /// Record how many token slots in `id` hold live KV data.
    ///
    /// `used` must be at most [`BlockPool::block_size`]. A freshly allocated
    /// block starts at zero, which counts as a fully wasted block until the
    /// caller fills it. Every owner of a shared block sees this same count.
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
    /// Free blocks contribute nothing. A shared block is counted once.
    pub fn wasted_slots(&self) -> u64 {
        self.refcounts
            .iter()
            .zip(self.used.iter())
            .filter(|(rc, _)| **rc > 0)
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
        match self.refcounts.get(id.0 as usize) {
            None => Err(AllocError::InvalidBlock(id)),
            Some(0) => Err(AllocError::NotAllocated(id)),
            Some(_) => Ok(()),
        }
    }

    /// Drop one owner of a block.
    ///
    /// The block returns to the free list only when its refcount reaches
    /// zero. Until then the initialized-token count and any stored KV
    /// vectors are left unchanged. At zero the vectors are cleared.
    pub fn free(&mut self, id: BlockId) -> Result<(), AllocError> {
        let index = id.0 as usize;
        let rc = self
            .refcounts
            .get_mut(index)
            .ok_or(AllocError::InvalidBlock(id))?;
        if *rc == 0 {
            return Err(AllocError::NotAllocated(id));
        }
        *rc -= 1;
        let dead = *rc == 0;
        if dead {
            self.used[index] = 0;
            self.zero_block(index);
            self.free_list.push(id.0);
        }
        Ok(())
    }

    /// Return a block the caller can write without aliasing other owners.
    ///
    /// A uniquely owned block is returned unchanged. A shared block is
    /// forked: this allocates a new block, copies the initialized-slot
    /// count onto it, copies any stored KV vectors for those slots, and
    /// drops one reference to `id`.
    ///
    /// On failure the original refcount, used-slot count, and KV vectors
    /// are unchanged.
    pub fn cow(&mut self, id: BlockId) -> Result<BlockId, AllocError> {
        let owners = self.refcount(id)?;
        if owners == 0 {
            return Err(AllocError::NotAllocated(id));
        }
        if owners == 1 {
            return Ok(id);
        }
        let used = self.used(id)?;
        let fresh = self.allocate()?;
        if let Err(err) = self.set_used(fresh, used) {
            let _ = self.free(fresh);
            return Err(err);
        }
        if let Err(err) = self.copy_kv(id, fresh, used) {
            let _ = self.free(fresh);
            return Err(err);
        }
        self.free(id)?;
        Ok(fresh)
    }

    /// Borrow the `f32` vector stored for one token slot.
    pub fn kv(&self, id: BlockId, slot: u32) -> Result<&[f32], AllocError> {
        let range = self.kv_range(id, slot)?;
        Ok(&self.kv.as_ref().expect("range requires storage").data[range])
    }

    /// Borrow the `f32` vector for a token slot this caller alone owns.
    ///
    /// A shared block returns [`AllocError::SharedWrite`]. Copy it with
    /// [`BlockPool::cow`] and write the new id.
    pub fn kv_mut(&mut self, id: BlockId, slot: u32) -> Result<&mut [f32], AllocError> {
        if self.refcount(id)? > 1 {
            return Err(AllocError::SharedWrite(id));
        }
        let range = self.kv_range(id, slot)?;
        Ok(&mut self.kv.as_mut().expect("range requires storage").data[range])
    }

    fn kv_range(&self, id: BlockId, slot: u32) -> Result<std::ops::Range<usize>, AllocError> {
        self.ensure_live(id)?;
        let Some(kv) = &self.kv else {
            return Err(AllocError::NoStorage);
        };
        if slot >= self.block_size {
            return Err(AllocError::InvalidSlot {
                block: id,
                slot,
                block_size: self.block_size,
            });
        }
        let width = kv.floats_per_token as usize;
        let start = (id.0 as usize * self.block_size as usize + slot as usize) * width;
        Ok(start..start + width)
    }

    fn zero_block(&mut self, index: usize) {
        let Some(kv) = self.kv.as_mut() else {
            return;
        };
        let width = self.block_size as usize * kv.floats_per_token as usize;
        let start = index * width;
        kv.data[start..start + width].fill(0.0);
    }

    fn copy_kv(&mut self, from: BlockId, to: BlockId, slots: u32) -> Result<(), AllocError> {
        if self.kv.is_none() || slots == 0 {
            return Ok(());
        }
        let width = self.kv.as_ref().expect("checked").floats_per_token as usize;
        let block = self.block_size as usize * width;
        let count = slots as usize * width;
        let src = from.0 as usize * block;
        let dst = to.0 as usize * block;
        let data = &mut self.kv.as_mut().expect("checked").data;
        let copied = data[src..src + count].to_vec();
        data[dst..dst + count].copy_from_slice(&copied);
        Ok(())
    }
}

/// Record of pool mutations made by one [`BlockTable::grow`], so a failed
/// grow can put the pool back.
enum GrowUndo {
    Allocated(BlockId),
    SetUsed { id: BlockId, prev: u32 },
    Cow { previous: BlockId, private: BlockId },
}

/// Ordered blocks backing one sequence.
///
/// The table borrows a [`BlockPool`] only for the duration of a method.
/// Each id it stores is one live reference. [`BlockTable::release`] drops
/// those references. Dropping a table without releasing leaks them for the
/// rest of the pool's life.
///
/// Blocks before the tail are full. The tail may be partial. Appending into
/// a shared tail copy-on-writes that block. Appending past a full block
/// allocates a new one, so a shared full block is never written in place.
#[derive(Debug)]
#[must_use = "call release to return block references to the pool"]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    len: u32,
}

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockTable {
    /// Empty table. It holds no block references.
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            len: 0,
        }
    }

    /// Logical tokens represented by this table.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// `true` when the table covers no tokens.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Blocks in token order. Only the last block may be partially used.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Where logical token `index` lives.
    ///
    /// `pool` must be the pool this table grew on. Its block size is the
    /// stride. The returned slot is what [`BlockPool::kv_mut`] writes.
    pub fn token_loc(&self, pool: &BlockPool, index: u32) -> Result<TokenLoc, AllocError> {
        if index >= self.len {
            return Err(AllocError::TokenOutOfRange {
                index,
                len: self.len,
            });
        }
        let block_size = pool.block_size();
        let block_index = (index / block_size) as usize;
        let Some(block) = self.blocks.get(block_index).copied() else {
            return Err(AllocError::TokenOutOfRange {
                index,
                len: self.len,
            });
        };
        Ok(TokenLoc {
            block,
            slot: index % block_size,
        })
    }

    /// Grow the table until it covers `new_len` tokens.
    ///
    /// When `new_len` crosses a block boundary, this allocates a new block.
    /// When the current tail is shared and the write extends into it, that
    /// tail is copy-on-write first.
    ///
    /// A failure leaves this table and the pool as they were. Shortening is
    /// rejected.
    pub fn grow(&mut self, pool: &mut BlockPool, new_len: u32) -> Result<(), AllocError> {
        if new_len < self.len {
            return Err(AllocError::LengthRegression {
                current: self.len,
                requested: new_len,
            });
        }
        if new_len == self.len {
            return Ok(());
        }

        let block_size = pool.block_size();
        let extra = BlockPool::blocks_needed(new_len, block_size)
            - BlockPool::blocks_needed(self.len, block_size);
        let mut cow_blocks: u32 = 0;
        if !self.len.is_multiple_of(block_size) {
            let tail = self.blocks[self.blocks.len() - 1];
            if pool.refcount(tail)? > 1 {
                cow_blocks = 1;
            }
        }
        let needed = u64::from(extra) + u64::from(cow_blocks);
        if u64::from(pool.free_count()) < needed {
            return Err(AllocError::Exhausted);
        }

        let snap_len = self.len;
        let snap_blocks = self.blocks.clone();
        let mut undo = Vec::new();
        while self.len < new_len {
            if let Err(err) = self.append_one(pool, &mut undo) {
                self.rollback_grow(pool, snap_len, snap_blocks, &mut undo);
                return Err(err);
            }
        }
        Ok(())
    }

    fn append_one(
        &mut self,
        pool: &mut BlockPool,
        undo: &mut Vec<GrowUndo>,
    ) -> Result<(), AllocError> {
        let block_size = pool.block_size();
        if self.len.is_multiple_of(block_size) {
            let id = pool.allocate()?;
            undo.push(GrowUndo::Allocated(id));
            pool.set_used(id, 1)?;
            self.blocks.push(id);
            self.len += 1;
            return Ok(());
        }

        let last = self.blocks.len() - 1;
        let tail = self.blocks[last];
        let used = pool.used(tail)?;
        if used >= block_size {
            return Err(AllocError::InvalidUsed {
                block: tail,
                used: used.saturating_add(1),
                block_size,
            });
        }
        if pool.refcount(tail)? > 1 {
            let private = pool.cow(tail)?;
            undo.push(GrowUndo::Cow {
                previous: tail,
                private,
            });
            self.blocks[last] = private;
            pool.set_used(private, used + 1)?;
            self.len += 1;
            return Ok(());
        }

        undo.push(GrowUndo::SetUsed {
            id: tail,
            prev: used,
        });
        pool.set_used(tail, used + 1)?;
        self.len += 1;
        Ok(())
    }

    fn rollback_grow(
        &mut self,
        pool: &mut BlockPool,
        len: u32,
        blocks: Vec<BlockId>,
        undo: &mut Vec<GrowUndo>,
    ) {
        while let Some(op) = undo.pop() {
            match op {
                GrowUndo::Allocated(id) => {
                    let _ = pool.free(id);
                }
                GrowUndo::SetUsed { id, prev } => {
                    let _ = pool.set_used(id, prev);
                }
                GrowUndo::Cow { previous, private } => {
                    let _ = pool.free(private);
                    let _ = pool.acquire(previous);
                }
            }
        }
        self.len = len;
        self.blocks = blocks;
    }

    /// Fork the first `prefix_len` tokens of this table.
    ///
    /// * A prefix that ends on a block boundary shares those full blocks.
    /// * A prefix that covers this table's entire length also shares a
    ///   partial tail. The next append copy-on-writes that tail.
    /// * A prefix that ends inside a block the parent has already written
    ///   past shares only the preceding full blocks. The child gets a new
    ///   private block holding the partial prefix, so it does not alias the
    ///   parent's later tokens.
    ///
    /// On failure, refcounts are unchanged and no new block stays allocated.
    pub fn fork_prefix(&self, pool: &mut BlockPool, prefix_len: u32) -> Result<Self, AllocError> {
        if prefix_len > self.len {
            return Err(AllocError::PrefixOutOfRange {
                prefix_len,
                len: self.len,
            });
        }
        if prefix_len == 0 {
            return Ok(Self::new());
        }

        let block_size = pool.block_size();
        let full_blocks = (prefix_len / block_size) as usize;
        let remainder = prefix_len % block_size;
        let share_partial = remainder > 0 && prefix_len == self.len;
        let share_count = full_blocks + usize::from(share_partial);

        let mut shared = Vec::with_capacity(share_count + usize::from(remainder > 0));
        for slot in 0..share_count {
            let id = self.blocks[slot];
            if let Err(err) = pool.acquire(id) {
                release_ids(pool, &shared);
                return Err(err);
            }
            shared.push(id);
        }

        if remainder > 0 && !share_partial {
            let id = match pool.allocate() {
                Ok(id) => id,
                Err(err) => {
                    release_ids(pool, &shared);
                    return Err(err);
                }
            };
            if let Err(err) = pool.set_used(id, remainder) {
                let _ = pool.free(id);
                release_ids(pool, &shared);
                return Err(err);
            }
            shared.push(id);
        }

        Ok(Self {
            blocks: shared,
            len: prefix_len,
        })
    }

    /// Drop one reference on every block in this table.
    ///
    /// Blocks that other tables still hold stay allocated. Later frees still
    /// run if an earlier one fails, and the first error is returned.
    pub fn release(self, pool: &mut BlockPool) -> Result<(), AllocError> {
        let mut first_err = None;
        for id in &self.blocks {
            if let Err(err) = pool.free(*id) {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

fn release_ids(pool: &mut BlockPool, ids: &[BlockId]) {
    for id in ids.iter().rev() {
        let _ = pool.free(*id);
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

    #[test]
    fn shared_block_stays_live_until_the_last_owner_frees_it() {
        let mut pool = BlockPool::new(1, 8);
        let id = pool.allocate().unwrap();
        pool.set_used(id, 8).unwrap();
        assert_eq!(pool.acquire(id).unwrap(), 2);
        assert_eq!(pool.refcount(id).unwrap(), 2);

        pool.free(id).unwrap();
        assert_eq!(pool.refcount(id).unwrap(), 1);
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.used(id).unwrap(), 8);

        pool.free(id).unwrap();
        assert_eq!(pool.refcount(id).unwrap(), 0);
        assert_eq!(pool.free_count(), 1);
        assert_eq!(pool.wasted_slots(), 0);
        assert_eq!(pool.used(id).unwrap_err(), AllocError::NotAllocated(id));
        assert_eq!(pool.free(id).unwrap_err(), AllocError::NotAllocated(id));
    }

    #[test]
    fn acquire_rejects_a_free_or_unknown_block() {
        let mut pool = BlockPool::new(1, 4);
        let id = pool.allocate().unwrap();
        pool.free(id).unwrap();
        assert_eq!(pool.acquire(id).unwrap_err(), AllocError::NotAllocated(id));
        assert_eq!(pool.free_count(), 1);

        let bogus = BlockId(5);
        assert_eq!(
            pool.acquire(bogus).unwrap_err(),
            AllocError::InvalidBlock(bogus)
        );
        assert_eq!(
            pool.refcount(bogus).unwrap_err(),
            AllocError::InvalidBlock(bogus)
        );
    }

    #[test]
    fn shared_owners_do_not_double_count_tail_waste() {
        let mut pool = BlockPool::new(1, 16);
        let id = pool.allocate().unwrap();
        pool.set_used(id, 10).unwrap();
        pool.acquire(id).unwrap();
        assert_eq!(pool.refcount(id).unwrap(), 2);
        assert_eq!(pool.wasted_slots(), 6);
        assert_eq!(pool.fragmentation(), 6.0 / 16.0);
    }

    #[test]
    fn cow_keeps_a_unique_block_in_place() {
        let mut pool = BlockPool::new(2, 8);
        let id = pool.allocate().unwrap();
        pool.set_used(id, 3).unwrap();
        assert_eq!(pool.cow(id).unwrap(), id);
        assert_eq!(pool.refcount(id).unwrap(), 1);
        assert_eq!(pool.used(id).unwrap(), 3);
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn cow_forks_a_shared_block_without_changing_the_other_owner() {
        let mut pool = BlockPool::new(2, 8);
        let shared = pool.allocate().unwrap();
        pool.set_used(shared, 5).unwrap();
        pool.acquire(shared).unwrap();

        let private = pool.cow(shared).unwrap();
        assert_ne!(private, shared);
        assert_eq!(pool.refcount(shared).unwrap(), 1);
        assert_eq!(pool.refcount(private).unwrap(), 1);
        assert_eq!(pool.used(shared).unwrap(), 5);
        assert_eq!(pool.used(private).unwrap(), 5);
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.wasted_slots(), 6);

        pool.set_used(private, 6).unwrap();
        assert_eq!(pool.used(shared).unwrap(), 5);
        assert_eq!(pool.used(private).unwrap(), 6);
    }

    #[test]
    fn cow_of_a_shared_block_fails_cleanly_when_the_pool_is_full() {
        let mut pool = BlockPool::new(1, 4);
        let shared = pool.allocate().unwrap();
        pool.set_used(shared, 4).unwrap();
        pool.acquire(shared).unwrap();

        assert_eq!(pool.cow(shared).unwrap_err(), AllocError::Exhausted);
        assert_eq!(pool.refcount(shared).unwrap(), 2);
        assert_eq!(pool.used(shared).unwrap(), 4);
        assert_eq!(pool.free_count(), 0);
    }

    #[test]
    fn stored_kv_is_writable_until_the_block_is_freed() {
        let mut pool = BlockPool::with_storage(2, 4, 3);
        assert_eq!(pool.floats_per_token(), Some(3));
        let id = pool.allocate().unwrap();
        pool.kv_mut(id, 1)
            .unwrap()
            .copy_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(pool.kv(id, 1).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(pool.kv(id, 0).unwrap(), &[0.0, 0.0, 0.0]);

        pool.free(id).unwrap();
        let again = pool.allocate().unwrap();
        assert_eq!(again, id);
        assert_eq!(pool.kv(again, 1).unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_metadata_pool_has_no_kv_vectors() {
        let mut pool = BlockPool::new(1, 4);
        let id = pool.allocate().unwrap();
        assert_eq!(pool.floats_per_token(), None);
        assert_eq!(pool.kv(id, 0).unwrap_err(), AllocError::NoStorage);
        assert_eq!(pool.kv_mut(id, 0).unwrap_err(), AllocError::NoStorage);
    }

    #[test]
    fn kv_rejects_a_slot_past_the_block_and_a_shared_write() {
        let mut pool = BlockPool::with_storage(1, 4, 2);
        let id = pool.allocate().unwrap();
        assert_eq!(
            pool.kv(id, 4).unwrap_err(),
            AllocError::InvalidSlot {
                block: id,
                slot: 4,
                block_size: 4,
            }
        );
        pool.acquire(id).unwrap();
        assert_eq!(pool.kv_mut(id, 0).unwrap_err(), AllocError::SharedWrite(id));
        assert_eq!(pool.kv(id, 0).unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn cow_copies_stored_kv_and_leaves_the_other_owner_unchanged() {
        let mut pool = BlockPool::with_storage(2, 4, 2);
        let shared = pool.allocate().unwrap();
        pool.set_used(shared, 1).unwrap();
        pool.kv_mut(shared, 0).unwrap().copy_from_slice(&[4.0, 5.0]);
        pool.acquire(shared).unwrap();

        let private = pool.cow(shared).unwrap();
        assert_eq!(pool.kv(shared, 0).unwrap(), &[4.0, 5.0]);
        assert_eq!(pool.kv(private, 0).unwrap(), &[4.0, 5.0]);
        assert_eq!(pool.kv(private, 1).unwrap(), &[0.0, 0.0]);

        pool.kv_mut(private, 0)
            .unwrap()
            .copy_from_slice(&[9.0, 8.0]);
        assert_eq!(pool.kv(shared, 0).unwrap(), &[4.0, 5.0]);
        assert_eq!(pool.kv(private, 0).unwrap(), &[9.0, 8.0]);
    }

    #[test]
    fn cow_rejects_a_free_block() {
        let mut pool = BlockPool::new(1, 4);
        let id = pool.allocate().unwrap();
        pool.free(id).unwrap();
        assert_eq!(pool.cow(id).unwrap_err(), AllocError::NotAllocated(id));
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn blocks_needed_rounds_up_a_partial_tail() {
        assert_eq!(BlockPool::blocks_needed(0, 4), 0);
        assert_eq!(BlockPool::blocks_needed(1, 4), 1);
        assert_eq!(BlockPool::blocks_needed(4, 4), 1);
        assert_eq!(BlockPool::blocks_needed(5, 4), 2);
    }

    #[test]
    fn grow_fills_full_blocks_and_a_partial_tail() {
        let mut pool = BlockPool::new(4, 4);
        let mut table = BlockTable::new();
        table.grow(&mut pool, 10).unwrap();
        assert_eq!(table.len(), 10);
        assert_eq!(table.blocks().len(), 3);
        assert_eq!(pool.used(table.blocks()[0]).unwrap(), 4);
        assert_eq!(pool.used(table.blocks()[1]).unwrap(), 4);
        assert_eq!(pool.used(table.blocks()[2]).unwrap(), 2);
        assert_eq!(pool.wasted_slots(), 2);
        assert_eq!(pool.free_count(), 1);

        table.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
        assert_eq!(pool.wasted_slots(), 0);
    }

    #[test]
    fn token_loc_maps_a_logical_index_onto_a_block_slot() {
        let mut pool = BlockPool::with_storage(4, 4, 2);
        let mut table = BlockTable::new();
        table.grow(&mut pool, 10).unwrap();

        assert_eq!(
            table.token_loc(&pool, 0).unwrap(),
            TokenLoc {
                block: table.blocks()[0],
                slot: 0,
            }
        );
        assert_eq!(
            table.token_loc(&pool, 6).unwrap(),
            TokenLoc {
                block: table.blocks()[1],
                slot: 2,
            }
        );
        let tail = table.token_loc(&pool, 9).unwrap();
        assert_eq!(tail.slot, 1);
        pool.kv_mut(tail.block, tail.slot)
            .unwrap()
            .copy_from_slice(&[1.5, 2.5]);
        assert_eq!(pool.kv(tail.block, tail.slot).unwrap(), &[1.5, 2.5]);
        assert_eq!(
            table.token_loc(&pool, 10).unwrap_err(),
            AllocError::TokenOutOfRange { index: 10, len: 10 }
        );

        table.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn fork_on_a_block_boundary_shares_only_the_covered_blocks() {
        let mut pool = BlockPool::new(4, 4);
        let mut parent = BlockTable::new();
        parent.grow(&mut pool, 10).unwrap();
        let child = parent.fork_prefix(&mut pool, 8).unwrap();

        assert_eq!(child.len(), 8);
        assert_eq!(child.blocks(), &parent.blocks()[..2]);
        assert_eq!(pool.refcount(parent.blocks()[0]).unwrap(), 2);
        assert_eq!(pool.refcount(parent.blocks()[1]).unwrap(), 2);
        assert_eq!(pool.refcount(parent.blocks()[2]).unwrap(), 1);

        let tail = parent.blocks()[2];
        parent.grow(&mut pool, 11).unwrap();
        assert_eq!(parent.blocks()[2], tail);
        assert_eq!(pool.used(tail).unwrap(), 3);
        assert_eq!(child.blocks().len(), 2);

        child.release(&mut pool).unwrap();
        parent.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn fork_of_the_whole_sequence_shares_the_tail_until_append() {
        let mut pool = BlockPool::new(4, 4);
        let mut parent = BlockTable::new();
        parent.grow(&mut pool, 10).unwrap();
        let tail = parent.blocks()[2];
        let mut child = parent.fork_prefix(&mut pool, 10).unwrap();
        assert_eq!(child.blocks(), parent.blocks());
        assert_eq!(pool.refcount(tail).unwrap(), 2);

        child.grow(&mut pool, 11).unwrap();
        assert_eq!(child.len(), 11);
        assert_ne!(child.blocks()[2], tail);
        assert_eq!(parent.blocks()[2], tail);
        assert_eq!(pool.refcount(tail).unwrap(), 1);
        assert_eq!(pool.used(tail).unwrap(), 2);
        assert_eq!(pool.used(child.blocks()[2]).unwrap(), 3);

        child.release(&mut pool).unwrap();
        parent.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn fork_inside_a_written_block_gets_a_private_boundary() {
        let mut pool = BlockPool::new(4, 4);
        let mut parent = BlockTable::new();
        parent.grow(&mut pool, 10).unwrap();
        let boundary = parent.blocks()[1];
        let mut child = parent.fork_prefix(&mut pool, 6).unwrap();

        assert_eq!(child.len(), 6);
        assert_eq!(child.blocks().len(), 2);
        assert_eq!(child.blocks()[0], parent.blocks()[0]);
        assert_ne!(child.blocks()[1], boundary);
        assert_eq!(pool.refcount(parent.blocks()[0]).unwrap(), 2);
        assert_eq!(pool.refcount(boundary).unwrap(), 1);
        assert_eq!(pool.used(boundary).unwrap(), 4);
        assert_eq!(pool.used(child.blocks()[1]).unwrap(), 2);

        child.grow(&mut pool, 7).unwrap();
        assert_eq!(pool.used(child.blocks()[1]).unwrap(), 3);
        assert_eq!(pool.used(boundary).unwrap(), 4);

        child.release(&mut pool).unwrap();
        parent.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn fork_rolls_back_acquires_when_the_private_block_cannot_be_allocated() {
        let mut pool = BlockPool::new(3, 4);
        let mut parent = BlockTable::new();
        parent.grow(&mut pool, 10).unwrap();
        assert_eq!(pool.free_count(), 0);

        let err = parent.fork_prefix(&mut pool, 6).unwrap_err();
        assert_eq!(err, AllocError::Exhausted);
        assert_eq!(pool.refcount(parent.blocks()[0]).unwrap(), 1);
        assert_eq!(pool.refcount(parent.blocks()[1]).unwrap(), 1);
        assert_eq!(pool.free_count(), 0);

        parent.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn extending_a_shared_tail_fails_cleanly_when_cow_cannot_allocate() {
        let mut pool = BlockPool::new(3, 4);
        let mut parent = BlockTable::new();
        parent.grow(&mut pool, 10).unwrap();
        let mut child = parent.fork_prefix(&mut pool, 10).unwrap();
        let tail = parent.blocks()[2];
        assert_eq!(pool.free_count(), 0);

        assert_eq!(
            child.grow(&mut pool, 11).unwrap_err(),
            AllocError::Exhausted
        );
        assert_eq!(child.len(), 10);
        assert_eq!(child.blocks()[2], tail);
        assert_eq!(pool.refcount(tail).unwrap(), 2);
        assert_eq!(pool.used(tail).unwrap(), 2);

        child.release(&mut pool).unwrap();
        parent.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn empty_and_overlong_prefixes_do_not_change_refcounts() {
        let mut pool = BlockPool::new(2, 4);
        let mut parent = BlockTable::new();
        parent.grow(&mut pool, 4).unwrap();

        let child = parent.fork_prefix(&mut pool, 0).unwrap();
        assert!(child.is_empty());
        child.release(&mut pool).unwrap();
        assert_eq!(pool.refcount(parent.blocks()[0]).unwrap(), 1);

        assert_eq!(
            parent.fork_prefix(&mut pool, 5).unwrap_err(),
            AllocError::PrefixOutOfRange {
                prefix_len: 5,
                len: 4,
            }
        );
        assert_eq!(pool.free_count(), 1);
        parent.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn grow_rejects_a_shorter_length_and_an_allocation_it_cannot_finish() {
        let mut pool = BlockPool::new(1, 4);
        let mut table = BlockTable::new();
        assert_eq!(table.grow(&mut pool, 8).unwrap_err(), AllocError::Exhausted);
        assert!(table.is_empty());
        assert_eq!(pool.free_count(), 1);

        table.grow(&mut pool, 3).unwrap();
        assert_eq!(
            table.grow(&mut pool, 1).unwrap_err(),
            AllocError::LengthRegression {
                current: 3,
                requested: 1,
            }
        );
        assert_eq!(table.len(), 3);
        table.grow(&mut pool, 3).unwrap();
        assert_eq!(table.blocks().len(), 1);

        table.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }
}
