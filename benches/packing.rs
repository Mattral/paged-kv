//! Sequences that fit, and slots wasted, under one fixed budget.
//!
//! Run with `cargo bench --bench packing`.

#[path = "support/packing_logic.rs"]
mod packing_logic;

use packing_logic::{measure, BLOCK_SIZE, BUDGET_SLOTS, LONG_LEN, MAX_LEN, SHORT_LEN};

fn main() {
    let fit = measure();
    println!("budget_slots={BUDGET_SLOTS}");
    println!("block_size={BLOCK_SIZE}");
    println!("short_len={SHORT_LEN}");
    println!("long_len={LONG_LEN}");
    println!("max_len={MAX_LEN}");
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
}
