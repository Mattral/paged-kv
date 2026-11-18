//! Sequences that fit when one prefix is shared, and when it is not.
//!
//! The slot budget and block size match the packing measurement. Every
//! sequence is a 512-token prefix plus a 48-token tail. The prefix ends on
//! a block boundary, so a hit shares those blocks and each tail takes new
//! ones.

use paged_kv::{BlockPool, BlockTable, PrefixCache};

/// Same slot budget as the packing measurement.
pub const BUDGET_SLOTS: u32 = 65_536;
/// Same block size as the packing measurement.
pub const BLOCK_SIZE: u32 = 16;
/// Shared prompt, in tokens. A multiple of [`BLOCK_SIZE`].
pub const PREFIX_LEN: u32 = 512;
/// Private tail, in tokens. A multiple of [`BLOCK_SIZE`].
pub const SUFFIX_LEN: u32 = 48;

const CACHE_KEY: u64 = 1;

/// How many sequences fit with and without sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixFit {
    /// Each sequence stores its own prefix.
    pub private: u32,
    /// One cached prefix, then a private tail per sequence.
    pub shared: u32,
}

/// Admit the same shape of sequence both ways, until the budget stops it.
pub fn measure() -> PrefixFit {
    PrefixFit {
        private: admit_private(),
        shared: admit_shared(),
    }
}

fn admit_private() -> u32 {
    let mut pool = BlockPool::new(BUDGET_SLOTS / BLOCK_SIZE, BLOCK_SIZE);
    let full = PREFIX_LEN + SUFFIX_LEN;
    let mut tables = Vec::new();
    loop {
        let mut table = BlockTable::new();
        if table.grow(&mut pool, full).is_err() {
            break;
        }
        tables.push(table);
    }
    let sequences = u32::try_from(tables.len()).expect("sequence count fits in u32");
    assert_eq!(pool.wasted_slots(), 0);
    for table in tables {
        table
            .release(&mut pool)
            .expect("each block was acquired once");
    }
    assert_eq!(pool.free_count(), pool.capacity());
    sequences
}

fn admit_shared() -> u32 {
    let mut pool = BlockPool::new(BUDGET_SLOTS / BLOCK_SIZE, BLOCK_SIZE);
    let mut prefix = BlockTable::new();
    prefix
        .grow(&mut pool, PREFIX_LEN)
        .expect("the prefix fits in the budget");
    let mut cache = PrefixCache::new();
    cache
        .insert(&mut pool, CACHE_KEY, &prefix)
        .expect("the cache keeps a fork of the prefix");
    prefix
        .release(&mut pool)
        .expect("the cache still holds the prefix");

    let full = PREFIX_LEN + SUFFIX_LEN;
    let mut tables = Vec::new();
    loop {
        let mut table = match cache.lookup(&mut pool, CACHE_KEY, PREFIX_LEN) {
            Ok(Some(table)) => table,
            Ok(None) => panic!("the prefix stays cached"),
            Err(_) => break,
        };
        if table.grow(&mut pool, full).is_err() {
            table
                .release(&mut pool)
                .expect("drop the fork that could not grow");
            break;
        }
        tables.push(table);
    }
    let sequences = u32::try_from(tables.len()).expect("sequence count fits in u32");
    assert_eq!(pool.wasted_slots(), 0);
    for table in tables {
        table
            .release(&mut pool)
            .expect("each tail was acquired once");
    }
    cache
        .clear(&mut pool)
        .expect("cached prefix blocks return to the pool");
    assert_eq!(pool.free_count(), pool.capacity());
    sequences
}
