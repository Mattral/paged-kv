//! Extra sequences that fit when one prefix is stored once.
//!
//! Run with `cargo bench --bench prefix_sharing`.

#[path = "support/prefix_logic.rs"]
mod prefix_logic;

use prefix_logic::{measure, BLOCK_SIZE, BUDGET_SLOTS, PREFIX_LEN, SUFFIX_LEN};

fn main() {
    let fit = measure();
    let extra = fit.shared - fit.private;
    println!("budget_slots={BUDGET_SLOTS}");
    println!("block_size={BLOCK_SIZE}");
    println!("prefix_len={PREFIX_LEN}");
    println!("suffix_len={SUFFIX_LEN}");
    println!("private_sequences={}", fit.private);
    println!("shared_sequences={}", fit.shared);
    println!("extra_sequences={extra}");
}
