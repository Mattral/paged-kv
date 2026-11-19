//! Sequences that fit, and slots wasted, under one fixed budget.
//!
//! Run with `cargo bench --bench packing`.

#[path = "support/packing_logic.rs"]
mod packing_logic;

use packing_logic::{
    measure, measure_synthetic, BLOCK_SIZE, BUDGET_SLOTS, TRACE_MAX_LEN, TRACE_REQUESTS,
};

fn main() {
    let fit = measure();
    println!("budget_slots={BUDGET_SLOTS}");
    println!("block_size={BLOCK_SIZE}");
    println!("trace=azure_llm_inference_2023_conversation");
    println!("requests={TRACE_REQUESTS}");
    println!("max_len={TRACE_MAX_LEN}");
    println!(
        "paged_sequences={} paged_wasted={}",
        fit.paged.sequences, fit.paged.wasted
    );
    println!(
        "doubling_sequences={} doubling_wasted={}",
        fit.doubling.sequences, fit.doubling.wasted
    );
    println!(
        "max_prealloc_sequences={} max_prealloc_wasted={}",
        fit.max_prealloc.sequences, fit.max_prealloc.wasted
    );

    let old = measure_synthetic();
    println!(
        "synthetic_paged_sequences={} synthetic_paged_wasted={}",
        old.paged.sequences, old.paged.wasted
    );
    println!(
        "synthetic_doubling_sequences={} synthetic_doubling_wasted={}",
        old.doubling.sequences, old.doubling.wasted
    );
    println!(
        "synthetic_max_prealloc_sequences={} synthetic_max_prealloc_wasted={}",
        old.max_prealloc.sequences, old.max_prealloc.wasted
    );
}
