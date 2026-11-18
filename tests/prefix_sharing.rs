#[path = "../benches/support/prefix_logic.rs"]
mod prefix_logic;

use prefix_logic::measure;

#[test]
fn a_shared_prefix_fits_more_sequences() {
    let fit = measure();
    assert!(fit.shared > fit.private);
    assert_eq!(fit.private, 117);
    assert_eq!(fit.shared, 1_354);
    assert_eq!(fit.shared - fit.private, 1_237);
}
