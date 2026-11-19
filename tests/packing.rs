#[path = "../benches/support/packing_logic.rs"]
mod packing_logic;

use packing_logic::{
    gib_thousandths, measure, measure_synthetic, slot_bytes, BUDGET_SLOTS, BYTES_PER_TOKEN,
};

#[test]
fn conversation_trace_fits_more_sequences_in_a_paged_pool() {
    let fit = measure();
    assert_eq!(BYTES_PER_TOKEN, 73_728);
    let budget_bytes = slot_bytes(u64::from(BUDGET_SLOTS));
    assert_eq!(budget_bytes, 4_831_838_208);
    assert_eq!(gib_thousandths(budget_bytes), 4_500);

    assert_eq!(fit.paged.sequences, 75);
    assert_eq!(fit.paged.wasted, 541);
    assert_eq!(slot_bytes(fit.paged.wasted), 39_886_848);
    assert_eq!(gib_thousandths(slot_bytes(fit.paged.wasted)), 37);

    assert_eq!(fit.doubling.sequences, 50);
    assert_eq!(fit.doubling.wasted, 24_496);
    assert_eq!(slot_bytes(fit.doubling.wasted), 1_806_041_088);
    assert_eq!(gib_thousandths(slot_bytes(fit.doubling.wasted)), 1_682);

    assert_eq!(fit.max_prealloc.sequences, 4);
    assert_eq!(fit.max_prealloc.wasted, 54_392);
    assert_eq!(slot_bytes(fit.max_prealloc.wasted), 4_010_213_376);
    assert_eq!(gib_thousandths(slot_bytes(fit.max_prealloc.wasted)), 3_735);
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
