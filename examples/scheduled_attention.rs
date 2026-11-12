//! Admit a hard-coded prompt and decode two tokens.
//!
//! The pool stores one vector per slot. Admission writes the prompt. Each
//! step writes one new token id, one past the last id, and attention reads
//! the stored K and V. That id rule is not a sampler. There is no tokenizer
//! and no model.
//!
//! Run with `cargo run --example scheduled_attention --quiet`.

use paged_kv::{BlockPool, Scheduler};

fn main() -> Result<(), paged_kv::AllocError> {
    let head_dim = 2u32;
    let prompt = [1u32, 2, 3, 4];
    let mut sched = Scheduler::new(BlockPool::with_storage(8, 4, head_dim * 2), 1);
    let id = sched.add_tokens(&prompt, 2);

    sched.admit()?;
    println!(
        "prefill {:?} scores {:?}",
        sched.token_ids(id).expect("prompt was written"),
        sched.scores(id).expect("prefill ran attention")
    );

    let mut steps = 0u32;
    loop {
        let step = sched.step()?;
        steps += 1;
        println!(
            "tokens {:?} scores {:?}",
            sched.token_ids(id).expect("decode wrote a token"),
            sched.scores(id).expect("decode ran attention")
        );
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
