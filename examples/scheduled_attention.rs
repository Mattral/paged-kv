//! Encode a short string and print the tokens the fixed-weight block predicts.
//!
//! Admission writes that block's K and V for the prompt. Each step appends
//! the id the block chose and writes that id's K and V. The block is not
//! trained.
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

    let ids = sched
        .token_ids(id)
        .expect("generation wrote tokens")
        .to_vec();
    assert_eq!(&ids[..prompt.len()], prompt.as_slice());
    assert_ne!(&ids[prompt.len()..], &[106, 107]);
    let scores = sched.scores(id).expect("scores stay until retire");
    let sum: f32 = scores.iter().copied().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    sched.retire()?;
    assert!(sched.is_idle());
    assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    println!("released; free blocks {}", sched.pool().free_count());
    Ok(())
}
