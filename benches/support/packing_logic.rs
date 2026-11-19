//! How many sequences fit in one slot budget.
//!
//! The schedule is the Azure conversation trace embedded in
//! [`TRACE_FILE`]. Each value is context tokens plus generated tokens,
//! in timestamp order. Admission stops at the first sequence that does
//! not fit. It does not skip ahead.
//!
//! [`measure_synthetic`] is the earlier repeating mix. The README keeps
//! those counts in a short note.
//!
//! Slots wasted are reserved slots with no live token. Slots never reserved
//! are not counted.

use paged_kv::{BlockPool, BlockTable, ContiguousTable};

/// Token slots shared by all three allocators.
pub const BUDGET_SLOTS: u32 = 65_536;
/// Block size for the paged pool.
pub const BLOCK_SIZE: u32 = 16;
/// Embedded conversation totals, one request per line.
const TRACE_FILE: &str = include_str!("azure_conv_totals.txt");
/// Requests in [`TRACE_FILE`].
pub const TRACE_REQUESTS: usize = 19_366;
/// Sum of the embedded totals.
pub const TRACE_TOKEN_SUM: u64 = 26_450_535;
/// Longest request in the embedded trace.
pub const TRACE_MAX_LEN: u32 = 14_089;

/// Length of a short sequence in the old synthetic mix.
pub const SHORT_LEN: u32 = 40;
/// Length of a long sequence in the old synthetic mix.
pub const LONG_LEN: u32 = 1_000;
/// Slots the synthetic mix reserved up front.
pub const SYNTHETIC_MAX_LEN: u32 = 1_024;

const SYNTHETIC_PERIOD: u32 = 16;

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
    /// One contiguous buffer per sequence, reserved to a fixed max length.
    pub max_prealloc: PolicyFit,
}

/// Context plus generated tokens, in trace order.
pub fn trace_lengths() -> Vec<u32> {
    let lengths: Vec<u32> = TRACE_FILE
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            line.parse::<u32>()
                .unwrap_or_else(|_| panic!("trace length is not a u32: {line}"))
        })
        .collect();
    assert_eq!(lengths.len(), TRACE_REQUESTS, "request count");
    assert_eq!(lengths.first().copied(), Some(418), "first request");
    assert_eq!(lengths.last().copied(), Some(380), "last request");
    let sum: u64 = lengths.iter().map(|len| u64::from(*len)).sum();
    assert_eq!(sum, TRACE_TOKEN_SUM, "token sum");
    assert_eq!(lengths.iter().copied().max(), Some(TRACE_MAX_LEN));
    lengths
}

/// Admit the conversation trace into each allocator until the budget stops it.
pub fn measure() -> Packing {
    let lengths = trace_lengths();
    let at = |index: u32| lengths.get(index as usize).copied();
    Packing {
        paged: paged(&at),
        doubling: doubling(&at),
        max_prealloc: max_prealloc(&at, TRACE_MAX_LEN),
    }
}

/// Admit the old repeating mix until the budget stops it.
pub fn measure_synthetic() -> Packing {
    let at = |index: u32| Some(synthetic_length_at(index));
    Packing {
        paged: paged(&at),
        doubling: doubling(&at),
        max_prealloc: max_prealloc(&at, SYNTHETIC_MAX_LEN),
    }
}

/// Length at `index` in the synthetic schedule.
///
/// Fifteen of every sixteen sequences are [`SHORT_LEN`] tokens. Every
/// sixteenth is [`LONG_LEN`].
pub fn synthetic_length_at(index: u32) -> u32 {
    if index % SYNTHETIC_PERIOD == SYNTHETIC_PERIOD - 1 {
        LONG_LEN
    } else {
        SHORT_LEN
    }
}

fn paged(len_at: &impl Fn(u32) -> Option<u32>) -> PolicyFit {
    let mut pool = BlockPool::new(BUDGET_SLOTS / BLOCK_SIZE, BLOCK_SIZE);
    let mut tables = Vec::new();
    let mut index = 0u32;
    loop {
        let Some(len) = len_at(index) else {
            break;
        };
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

fn doubling(len_at: &impl Fn(u32) -> Option<u32>) -> PolicyFit {
    admit_contiguous(len_at, |len| {
        let mut table = ContiguousTable::new();
        table.grow(len).expect("length only grows");
        table.capacity()
    })
}

fn max_prealloc(len_at: &impl Fn(u32) -> Option<u32>, max_len: u32) -> PolicyFit {
    admit_contiguous(len_at, |len| {
        let mut table = ContiguousTable::new();
        table.reserve_exact(max_len);
        table.grow(len).expect("the reserve covers the length");
        table.capacity()
    })
}

fn admit_contiguous(
    len_at: &impl Fn(u32) -> Option<u32>,
    reserve: impl Fn(u32) -> u32,
) -> PolicyFit {
    let mut reserved = 0u64;
    let mut wasted = 0u64;
    let mut sequences = 0u32;
    let mut index = 0u32;
    loop {
        let Some(len) = len_at(index) else {
            break;
        };
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
