#[path = "../benches/support/packing_logic.rs"]
mod packing_logic;

use packing_logic::{measure, measure_synthetic};

#[test]
fn conversation_trace_fits_more_sequences_in_a_paged_pool() {
    let fit = measure();
    assert_eq!(fit.paged.sequences, 75);
    assert_eq!(fit.paged.wasted, 541);
    assert_eq!(fit.doubling.sequences, 50);
    assert_eq!(fit.doubling.wasted, 24_496);
    assert_eq!(fit.max_prealloc.sequences, 4);
    assert_eq!(fit.max_prealloc.wasted, 54_392);
}

#[test]
fn synthetic_mix_fits_607_529_and_64() {
    let fit = measure_synthetic();
    assert_eq!(fit.paged.sequences, 607);
    assert_eq!(fit.paged.wasted, 4_856);
    assert_eq!(fit.doubling.sequences, 529);
    assert_eq!(fit.doubling.wasted, 12_696);
    assert_eq!(fit.max_prealloc.sequences, 64);
    assert_eq!(fit.max_prealloc.wasted, 59_136);
}
