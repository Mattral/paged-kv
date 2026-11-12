//! Encode a short string and decode the tokens the scheduler writes.
//!
//! The pool stores one vector per slot. Admission writes the prompt. Each
//! step still appends one past the last token id. There is no model.
//!
//! Run with `cargo run --example scheduled_attention --quiet`.

use paged_kv::{decode, encode, BlockPool, Scheduler};

fn main() -> Result<(), paged_kv::AllocError> {
    let head_dim = 2u32;
    let text = "hi";
    let prompt = encode(text);
    let mut sched = Scheduler::new(BlockPool::with_storage(8, 4, head_dim * 2), 1);
    let id = sched.add_tokens(&prompt, 2);

    sched.admit()?;
    let ids = sched.token_ids(id).expect("prompt was written");
    println!("prompt {text:?} -> {ids:?} ({:?})", decode(ids));

    let mut steps = 0u32;
    loop {
        let step = sched.step()?;
        steps += 1;
        let ids = sched.token_ids(id).expect("decode wrote a token");
        println!("tokens {ids:?} text {:?}", decode(ids));
        if step.finished.contains(&id) {
            break;
        }
        assert!(steps < 8, "generation did not finish");
    }

    let scores = sched.scores(id).expect("scores stay until retire").to_vec();
    assert_eq!(scores.len(), prompt.len() + 2);
    let sum: f32 = scores.iter().copied().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    sched.retire()?;
    assert!(sched.is_idle());
    assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    println!("released; free blocks {}", sched.pool().free_count());
    Ok(())
}
