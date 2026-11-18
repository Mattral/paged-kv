#[path = "../benches/support/packing_logic.rs"]
mod packing_logic;

use packing_logic::measure;

#[test]
fn skewed_lengths_fit_more_sequences_in_a_paged_pool() {
    let fit = measure();
    assert_eq!(fit.paged.sequences, 607);
    assert_eq!(fit.paged.wasted, 4_856);
    assert_eq!(fit.doubling.sequences, 529);
    assert_eq!(fit.doubling.wasted, 12_696);
    assert_eq!(fit.max_prealloc.sequences, 64);
    assert_eq!(fit.max_prealloc.wasted, 59_136);
}
