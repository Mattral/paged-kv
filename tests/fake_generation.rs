//! A model-free generation loop.
//!
//! Each step admits from the waiting queue, appends one token to every
//! decoding sequence, and retires whatever finished. No tensors are read.

use paged_kv::{BlockPool, Phase, Scheduler};

#[test]
fn fake_generation_joins_sequences_and_frees_every_block() {
    let mut sched = Scheduler::new(BlockPool::new(8, 4), 2);
    let first = sched.add(4, 2);
    let second = sched.add(4, 1);
    let third = sched.add(8, 2);

    let mut decoded = 0usize;
    let mut saw_third_waiting_behind_a_full_batch = false;
    let mut late = None;
    let mut late_waited_while_third_was_running = false;

    for tick in 0..32 {
        if sched.is_idle() && late.is_some() {
            break;
        }
        if tick == 1 {
            late = Some(sched.add(4, 1));
        }

        sched.admit().unwrap();
        if sched.running().len() == 2 && sched.next_waiting().map(|seq| seq.id()) == Some(third) {
            saw_third_waiting_behind_a_full_batch = true;
        }
        if let Some(late_id) = late {
            let late_is_running = sched.running().iter().any(|seq| seq.id() == late_id);
            let third_still_waiting = sched.next_waiting().map(|seq| seq.id()) == Some(third);
            assert!(
                !(late_is_running && third_still_waiting),
                "a later arrival ran ahead of an older waiting prompt"
            );
            if sched.next_waiting().map(|seq| seq.id()) == Some(late_id)
                && sched.running().iter().any(|seq| seq.id() == third)
            {
                late_waited_while_third_was_running = true;
            }
        }

        if sched.running().iter().any(|seq| seq.phase() == Phase::Decoding) {
            decoded += sched.step().unwrap().decoded.len();
        }
        sched.retire().unwrap();
    }

    assert!(sched.is_idle(), "generation loop stalled");
    assert!(saw_third_waiting_behind_a_full_batch);
    assert!(late_waited_while_third_was_running);
    assert_eq!(decoded, 2 + 1 + 2 + 1);
    assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    let _ = (first, second);
}
