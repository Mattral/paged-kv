//! Scheduler batching versus a one-sequence-at-a-time loop.
//!
//! Both paths reserve the same prompts and append the same number of decode
//! tokens through [`BlockPool`]. The naive loop finishes each sequence before
//! starting the next. The scheduler keeps `BATCH` sequences live and admits
//! the next arrival when a slot frees.
//!
//! This measures bookkeeping only. There is no model and no tensor copy.

use std::hint::black_box;
use std::time::Instant;

use paged_kv::{BlockPool, BlockTable, Phase, Scheduler};

const SEQUENCES: u32 = 256;
const PROMPT: u32 = 128;
const NEW_TOKENS: u32 = 64;
const BLOCK_SIZE: u32 = 16;
const BATCH: usize = 16;
const WARMUP: usize = 3;
const ITERS: usize = 21;

fn pool_blocks() -> u32 {
    let per_seq = BlockPool::blocks_needed(PROMPT + NEW_TOKENS, BLOCK_SIZE);
    per_seq * u32::try_from(BATCH).expect("batch fits in u32")
}

fn run_scheduler() -> u32 {
    let mut sched = Scheduler::new(BlockPool::new(pool_blocks(), BLOCK_SIZE), BATCH);
    for _ in 0..SEQUENCES {
        sched.add(PROMPT, NEW_TOKENS);
    }
    let mut decoded = 0u32;
    while !sched.is_idle() {
        sched.admit().expect("prompt fits the batch pool");
        if sched
            .running()
            .iter()
            .any(|seq| seq.phase() == Phase::Decoding)
        {
            let step = sched.step().expect("batch was checked before decode");
            decoded += u32::try_from(step.decoded.len()).expect("decoded count fits in u32");
        }
        sched.retire().expect("finished blocks release");
    }
    assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    decoded
}

fn run_naive() -> u32 {
    let mut pool = BlockPool::new(pool_blocks(), BLOCK_SIZE);
    let mut decoded = 0u32;
    for _ in 0..SEQUENCES {
        let mut table = BlockTable::new();
        table
            .grow(&mut pool, PROMPT)
            .expect("prompt fits the pool");
        let mut len = PROMPT;
        for _ in 0..NEW_TOKENS {
            len += 1;
            table.grow(&mut pool, len).expect("token fits the pool");
            decoded += 1;
        }
        table.release(&mut pool).expect("blocks release");
    }
    assert_eq!(pool.free_count(), pool.capacity());
    decoded
}

fn samples(run: fn() -> u32) -> Vec<u128> {
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        black_box(run());
        times.push(start.elapsed().as_nanos());
    }
    times
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
    println!("{name}_ns_min={min}");
    println!("{name}_ns_median={med}");
    println!("{name}_ns_max={max}");
}

fn main() {
    for _ in 0..WARMUP {
        black_box(run_naive());
        black_box(run_scheduler());
    }
    let naive = samples(run_naive);
    let scheduler = samples(run_scheduler);
    let naive_med = median(&naive);
    let scheduler_med = median(&scheduler);
    let ratio = scheduler_med as f64 / naive_med as f64;

    println!("sequences={SEQUENCES}");
    println!("prompt_tokens={PROMPT}");
    println!("new_tokens={NEW_TOKENS}");
    println!("block_size={BLOCK_SIZE}");
    println!("batch={BATCH}");
    println!("pool_blocks={}", pool_blocks());
    println!("warmup={WARMUP}");
    println!("iters={ITERS}");
    report("naive", &naive);
    report("scheduler", &scheduler);
    println!("scheduler_over_naive_median={ratio:.3}");
}
