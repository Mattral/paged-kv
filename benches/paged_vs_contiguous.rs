//! Paged blocks versus one contiguous span per sequence.
//!
//! Four policies make room for the same tokens:
//!
//! * paged, one token at a time, which is how decode calls `grow`
//! * contiguous doubling, one token at a time
//! * paged, one `grow` to the full length
//! * contiguous `reserve_exact`, then one `grow`
//!
//! Every sequence stays live together. The pool is sized for that. No KV
//! bytes are stored, and prefixes are not shared.

use std::hint::black_box;
use std::time::Instant;

use paged_kv::{BlockPool, BlockTable, ContiguousTable};

const SEQUENCES: u32 = 512;
const TOKENS: u32 = 260;
const BLOCK_SIZE: u32 = 16;
const WARMUP: usize = 3;
const ITERS: usize = 21;

fn pool_blocks() -> u32 {
    BlockPool::blocks_needed(TOKENS, BLOCK_SIZE) * SEQUENCES
}

fn paged_token_by_token() -> u128 {
    let mut pool = BlockPool::new(pool_blocks(), BLOCK_SIZE);
    let start = Instant::now();
    let mut tables = Vec::with_capacity(SEQUENCES as usize);
    for _ in 0..SEQUENCES {
        let mut table = BlockTable::new();
        for len in 1..=TOKENS {
            table
                .grow(&mut pool, len)
                .expect("pool fits every sequence");
        }
        tables.push(table);
    }
    let elapsed = start.elapsed().as_nanos();
    let live = pool.capacity() - pool.free_count();
    for table in tables {
        table
            .release(&mut pool)
            .expect("each block was acquired once");
    }
    assert_eq!(pool.free_count(), pool.capacity());
    black_box(live);
    elapsed
}

fn contiguous_doubling() -> u128 {
    let start = Instant::now();
    let mut tables = Vec::with_capacity(SEQUENCES as usize);
    for _ in 0..SEQUENCES {
        let mut table = ContiguousTable::new();
        for len in 1..=TOKENS {
            table.grow(len).expect("length only grows");
        }
        tables.push(table);
    }
    let elapsed = start.elapsed().as_nanos();
    let reserved: u64 = tables.iter().map(|table| u64::from(table.capacity())).sum();
    black_box(reserved);
    elapsed
}

fn paged_one_shot() -> u128 {
    let mut pool = BlockPool::new(pool_blocks(), BLOCK_SIZE);
    let start = Instant::now();
    let mut tables = Vec::with_capacity(SEQUENCES as usize);
    for _ in 0..SEQUENCES {
        let mut table = BlockTable::new();
        table
            .grow(&mut pool, TOKENS)
            .expect("pool fits every sequence");
        tables.push(table);
    }
    let elapsed = start.elapsed().as_nanos();
    let live = pool.capacity() - pool.free_count();
    for table in tables {
        table
            .release(&mut pool)
            .expect("each block was acquired once");
    }
    assert_eq!(pool.free_count(), pool.capacity());
    black_box(live);
    elapsed
}

fn contiguous_exact() -> u128 {
    let start = Instant::now();
    let mut tables = Vec::with_capacity(SEQUENCES as usize);
    for _ in 0..SEQUENCES {
        let mut table = ContiguousTable::new();
        table.reserve_exact(TOKENS);
        table
            .grow(TOKENS)
            .expect("reserved capacity covers the length");
        tables.push(table);
    }
    let elapsed = start.elapsed().as_nanos();
    let reserved: u64 = tables.iter().map(|table| u64::from(table.capacity())).sum();
    black_box(reserved);
    elapsed
}

fn median(times: &[u128]) -> u128 {
    let mut ordered = times.to_vec();
    ordered.sort_unstable();
    ordered[ordered.len() / 2]
}

fn collect(sample: fn() -> u128) -> Vec<u128> {
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        times.push(sample());
    }
    times
}

fn report(name: &str, times: &[u128]) {
    let med = median(times);
    let min = times.iter().copied().min().expect("at least one sample");
    let max = times.iter().copied().max().expect("at least one sample");
    println!("{name}_ns_min={min}");
    println!("{name}_ns_median={med}");
    println!("{name}_ns_max={max}");
}

fn main() {
    for _ in 0..WARMUP {
        black_box(paged_token_by_token());
        black_box(contiguous_doubling());
        black_box(paged_one_shot());
        black_box(contiguous_exact());
    }
    println!("sequences={SEQUENCES}");
    println!("tokens={TOKENS}");
    println!("block_size={BLOCK_SIZE}");
    println!("pool_blocks={}", pool_blocks());
    println!("warmup={WARMUP}");
    println!("iters={ITERS}");
    let paged_step = collect(paged_token_by_token);
    let doubling = collect(contiguous_doubling);
    let paged_once = collect(paged_one_shot);
    let exact = collect(contiguous_exact);
    report("paged_token", &paged_step);
    report("contiguous_doubling", &doubling);
    report("paged_one_shot", &paged_once);
    report("contiguous_exact", &exact);

    let mut doubled = ContiguousTable::new();
    for len in 1..=TOKENS {
        doubled.grow(len).unwrap();
    }
    let paged_slots = u64::from(pool_blocks()) * u64::from(BLOCK_SIZE);
    let logical = u64::from(SEQUENCES) * u64::from(TOKENS);
    println!("logical_slots={logical}");
    println!("paged_reserved_slots={paged_slots}");
    println!(
        "doubling_reserved_slots={}",
        u64::from(doubled.capacity()) * u64::from(SEQUENCES)
    );
    println!("exact_reserved_slots={logical}");
}
