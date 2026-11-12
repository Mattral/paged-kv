//! Fill a few token slots, score the stored vectors, and show a key change.
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

    let before = attend(&pool, &table, head_dim)?;
    println!("scores: {:?}", before.scores);
    println!("context: {:?}", before.context);

    let first = table.token_loc(&pool, 0)?;
    let mut row = pool.kv(first.block, first.slot)?.to_vec();
    for value in &mut row[..head_dim as usize] {
        *value = 100.0;
    }
    pool.kv_mut(first.block, first.slot)?.copy_from_slice(&row);
    let after = attend(&pool, &table, head_dim)?;
    println!("scores after raising the first key: {:?}", after.scores);
    assert!(after.scores[0] > before.scores[0]);

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
