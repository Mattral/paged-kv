//! Block allocate and free under a full pool.
//!
//! Each sample fills every block, marks it fully used, then frees every
//! block. Pool construction is outside the timer. The reuse sample does one
//! untimed fill and drain first, then times the second pass.
//!
//! This is metadata only. No KV bytes are written.

use std::hint::black_box;
use std::time::Instant;

use paged_kv::{BlockId, BlockPool};

const BLOCKS: u32 = 100_000;
const BLOCK_SIZE: u32 = 16;
const WARMUP: usize = 3;
const ITERS: usize = 21;

fn fill(pool: &mut BlockPool) -> Vec<BlockId> {
    let mut ids = Vec::with_capacity(pool.capacity() as usize);
    for _ in 0..pool.capacity() {
        let id = pool.allocate().expect("pool has a free block");
        pool.set_used(id, BLOCK_SIZE)
            .expect("fresh block accepts a full write");
        ids.push(id);
    }
    ids
}

fn drain(pool: &mut BlockPool, ids: &[BlockId]) {
    for id in ids {
        pool.free(*id).expect("block is still live");
    }
}

fn sample_fresh() -> u128 {
    let mut pool = BlockPool::new(BLOCKS, BLOCK_SIZE);
    let start = Instant::now();
    let ids = fill(&mut pool);
    drain(&mut pool, &ids);
    let elapsed = start.elapsed().as_nanos();
    assert_eq!(pool.free_count(), pool.capacity());
    black_box(ids);
    elapsed
}

fn sample_reuse() -> u128 {
    let mut pool = BlockPool::new(BLOCKS, BLOCK_SIZE);
    let ids = fill(&mut pool);
    drain(&mut pool, &ids);
    let start = Instant::now();
    let ids = fill(&mut pool);
    drain(&mut pool, &ids);
    let elapsed = start.elapsed().as_nanos();
    assert_eq!(pool.free_count(), pool.capacity());
    black_box(ids);
    elapsed
}

fn median(times: &[u128]) -> u128 {
    let mut ordered = times.to_vec();
    ordered.sort_unstable();
    ordered[ordered.len() / 2]
}

fn report(name: &str, times: &[u128]) {
    let med = median(times);
    let min = times.iter().copied().min().expect("at least one sample");
    let max = times.iter().copied().max().expect("at least one sample");
    let per_block = med as f64 / f64::from(BLOCKS);
    println!("{name}_ns_min={min}");
    println!("{name}_ns_median={med}");
    println!("{name}_ns_max={max}");
    println!("{name}_ns_per_block_median={per_block:.2}");
}

fn collect(sample: fn() -> u128) -> Vec<u128> {
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        times.push(sample());
    }
    times
}

fn main() {
    for _ in 0..WARMUP {
        black_box(sample_fresh());
        black_box(sample_reuse());
    }
    let fresh = collect(sample_fresh);
    let reuse = collect(sample_reuse);
    println!("blocks={BLOCKS}");
    println!("block_size={BLOCK_SIZE}");
    println!("warmup={WARMUP}");
    println!("iters={ITERS}");
    report("fresh", &fresh);
    report("reuse", &reuse);
    let ratio = median(&reuse) as f64 / median(&fresh) as f64;
    println!("reuse_over_fresh_median={ratio:.3}");
}
