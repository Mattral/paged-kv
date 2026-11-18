//! How many sequences fit in one slot budget.
//!
//! Fifteen of every sixteen sequences are [`SHORT_LEN`] tokens. Every
//! sixteenth is [`LONG_LEN`]. Admission stops at the first sequence that
//! does not fit. It does not skip ahead.
//!
//! Slots wasted are reserved slots with no live token. Slots never reserved
//! are not counted.

use paged_kv::{BlockPool, BlockTable, ContiguousTable};

/// Token slots shared by all three allocators.
pub const BUDGET_SLOTS: u32 = 65_536;
/// Block size for the paged pool.
pub const BLOCK_SIZE: u32 = 16;
/// Length of a short sequence.
pub const SHORT_LEN: u32 = 40;
/// Length of a long sequence.
pub const LONG_LEN: u32 = 1_000;
/// Slots reserved up front by max-length preallocation.
pub const MAX_LEN: u32 = 1_024;

const PERIOD: u32 = 16;

/// One allocator's packing result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyFit {
    /// Admitted sequences.
    pub sequences: u32,
    /// Reserved slots that do not hold a live token.
    pub wasted: u64,
}

/// The three allocators, in comparison order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Packing {
    /// This crate's block pool.
    pub paged: PolicyFit,
    /// One contiguous buffer per sequence, rounded up by doubling.
    pub doubling: PolicyFit,
    /// One contiguous buffer per sequence, reserved to [`MAX_LEN`].
    pub max_prealloc: PolicyFit,
}

/// Length at `index` in the skewed schedule.
pub fn length_at(index: u32) -> u32 {
    if index % PERIOD == PERIOD - 1 {
        LONG_LEN
    } else {
        SHORT_LEN
    }
}

/// Admit the skewed schedule into each allocator until the budget stops it.
pub fn measure() -> Packing {
    Packing {
        paged: paged(),
        doubling: doubling(),
        max_prealloc: max_prealloc(),
    }
}

fn paged() -> PolicyFit {
    let mut pool = BlockPool::new(BUDGET_SLOTS / BLOCK_SIZE, BLOCK_SIZE);
    let mut tables = Vec::new();
    let mut index = 0u32;
    loop {
        let len = length_at(index);
        let mut table = BlockTable::new();
        if table.grow(&mut pool, len).is_err() {
            break;
        }
        tables.push(table);
        index += 1;
    }
    let sequences = u32::try_from(tables.len()).expect("sequence count fits in u32");
    let wasted = pool.wasted_slots();
    let reserved = u64::from(pool.capacity() - pool.free_count()) * u64::from(BLOCK_SIZE);
    let live: u64 = tables.iter().map(|table| u64::from(table.len())).sum();
    assert_eq!(wasted, reserved - live);
    for table in tables {
        table
            .release(&mut pool)
            .expect("each block was acquired once");
    }
    assert_eq!(pool.free_count(), pool.capacity());
    PolicyFit { sequences, wasted }
}

fn doubling() -> PolicyFit {
    admit_contiguous(|len| {
        let mut table = ContiguousTable::new();
        table.grow(len).expect("length only grows");
        table.capacity()
    })
}

fn max_prealloc() -> PolicyFit {
    admit_contiguous(|len| {
        let mut table = ContiguousTable::new();
        table.reserve_exact(MAX_LEN);
        table.grow(len).expect("the reserve covers the length");
        table.capacity()
    })
}

fn admit_contiguous(reserve: impl Fn(u32) -> u32) -> PolicyFit {
    let mut reserved = 0u64;
    let mut wasted = 0u64;
    let mut sequences = 0u32;
    let mut index = 0u32;
    loop {
        let len = length_at(index);
        let cap = u64::from(reserve(len));
        if reserved + cap > u64::from(BUDGET_SLOTS) {
            break;
        }
        reserved += cap;
        wasted += cap - u64::from(len);
        sequences += 1;
        index += 1;
    }
    PolicyFit { sequences, wasted }
}
