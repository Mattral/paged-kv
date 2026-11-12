//! Fill a few token slots and score them.
//!
//! Run with `cargo run --example fake_attention --quiet`.

use paged_kv::{attend, fill_tokens, BlockPool, BlockTable};

fn main() -> Result<(), paged_kv::AllocError> {
    let head_dim = 4u32;
    let tokens = [7u32, 3, 9, 1, 4];
    let mut pool = BlockPool::with_storage(4, 4, head_dim * 2);
    let mut table = BlockTable::new();

    fill_tokens(&mut pool, &mut table, head_dim, &tokens)?;
    for (index, token) in tokens.iter().copied().enumerate() {
        let loc = table.token_loc(&pool, index as u32)?;
        println!(
            "token {token} -> block {} slot {}",
            loc.block.index(),
            loc.slot
        );
    }

    let scores = attend(&pool, &table, head_dim, *tokens.last().unwrap())?;
    println!(
        "scores against token {}: {scores:?}",
        tokens.last().unwrap()
    );

    let tail = *table.blocks().last().expect("fill allocated a block");
    table.release(&mut pool)?;
    let reused = pool.allocate()?;
    let slot = pool.kv(reused, 0)?;
    println!(
        "released the table, allocated block {} again, slot 0 is {slot:?}",
        reused.index()
    );
    assert_eq!(reused, tail);
    assert!(slot.iter().all(|value| *value == 0.0));
    pool.free(reused)?;
    Ok(())
}
