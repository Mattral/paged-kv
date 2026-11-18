//! Print tokens from `openai-community/gpt2`.
//!
//! The first run downloads `model.safetensors` and `tokenizer.json` into the
//! Hugging Face cache. Admission writes that model's K and V for the prompt.
//! Each step appends the id from its logits and writes that token's K and V.
//! One sequence runs. This is not a model server.
//!
//! ```sh
//! cargo run --example scheduled_attention -- "The capital of France is"
//! ```

use paged_kv::{model_decode, model_encode, BlockPool, Scheduler, KV_FLOATS};

fn main() -> Result<(), paged_kv::AllocError> {
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The capital of France is".to_string());
    let max_new = 8u32;
    let prompt = model_encode(&text)?;
    let prompt_len = u32::try_from(prompt.len()).expect("prompt length fits in u32");
    let total = prompt_len + max_new;
    let blocks = BlockPool::blocks_needed(total, 16).max(1);
    let mut sched = Scheduler::new(BlockPool::with_storage(blocks, 16, KV_FLOATS), 1);
    let id = sched.add_tokens(&prompt, max_new);

    sched.admit()?;
    let ids = sched.token_ids(id).expect("prompt was written");
    println!("prompt {text:?} -> {ids:?} ({:?})", model_decode(ids)?);

    let mut steps = 0u32;
    loop {
        let step = sched.step()?;
        steps += 1;
        let ids = sched.token_ids(id).expect("decode wrote a token");
        println!("tokens {ids:?} text {:?}", model_decode(ids)?);
        if step.finished.contains(&id) {
            break;
        }
        assert!(steps < max_new, "generation did not finish");
    }

    let ids = sched
        .token_ids(id)
        .expect("generation wrote tokens")
        .to_vec();
    assert_eq!(&ids[..prompt.len()], prompt.as_slice());
    assert!(ids.len() > prompt.len());
    let scores = sched.scores(id).expect("scores stay until retire");
    let sum: f32 = scores.iter().copied().sum();
    assert!((sum - 1.0).abs() < 1e-4);

    sched.retire()?;
    assert!(sched.is_idle());
    assert_eq!(sched.pool().free_count(), sched.pool().capacity());
    println!("released; free blocks {}", sched.pool().free_count());
    Ok(())
}
