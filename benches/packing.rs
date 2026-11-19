//! Sequences that fit, and slots wasted, under one fixed budget.
//!
//! Run with `cargo bench --bench packing`.

#[path = "support/packing_logic.rs"]
mod packing_logic;

use packing_logic::{
    gib_thousandths, measure, measure_synthetic, slot_bytes, BLOCK_SIZE, BUDGET_SLOTS,
    BYTES_PER_TOKEN, TRACE_MAX_LEN, TRACE_REQUESTS,
};

fn main() {
    let fit = measure();
    println!("budget_slots={BUDGET_SLOTS}");
    println!("block_size={BLOCK_SIZE}");
    println!("trace=azure_llm_inference_2023_conversation");
    println!("requests={TRACE_REQUESTS}");
    println!("max_len={TRACE_MAX_LEN}");
    println!("bytes_per_token={BYTES_PER_TOKEN}");
    let budget_bytes = slot_bytes(u64::from(BUDGET_SLOTS));
    let budget_gib = gib_thousandths(budget_bytes);
    println!(
        "budget_bytes={budget_bytes} budget_gib={}.{:03}",
        budget_gib / 1000,
        budget_gib % 1000
    );
    print_fit("paged", fit.paged.sequences, fit.paged.wasted);
    print_fit("doubling", fit.doubling.sequences, fit.doubling.wasted);
    print_fit(
        "max_prealloc",
        fit.max_prealloc.sequences,
        fit.max_prealloc.wasted,
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

fn print_fit(name: &str, sequences: u32, wasted: u64) {
    let bytes = slot_bytes(wasted);
    let gib = gib_thousandths(bytes);
    println!(
        "{name}_sequences={sequences} {name}_wasted={wasted} {name}_wasted_bytes={bytes} {name}_wasted_gib={}.{:03}",
        gib / 1000,
        gib % 1000
    );
}
