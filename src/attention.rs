//! One attention step over KV vectors stored in a [`BlockPool`](crate::BlockPool).
//!
//! K and V are a fixed function of a token id. There is no learned weight
//! and no tokenizer. [`fill_tokens`] allocates blocks for those ids and
//! writes the vectors. [`attend`] reads K back and returns a dot product
//! per token. V is stored so a later step can see it. It is not multiplied
//! in here.

use crate::{AllocError, BlockPool, BlockTable, TokenLoc};

/// Allocate blocks for `token_ids` and write each token's K and V.
///
/// The table grows by one token at a time, which is the decode shape.
/// `head_dim * 2` must equal the pool's stored width.
pub fn fill_tokens(
    pool: &mut BlockPool,
    table: &mut BlockTable,
    head_dim: u32,
    token_ids: &[u32],
) -> Result<(), AllocError> {
    for token_id in token_ids.iter().copied() {
        let new_len = table
            .len()
            .checked_add(1)
            .expect("token length fits in u32");
        table.grow(pool, new_len)?;
        let loc = table.token_loc(pool, new_len - 1)?;
        write_token(pool, loc, head_dim, token_id)?;
    }
    Ok(())
}

/// Write K and V for `token_id` into `loc`.
///
/// The first `head_dim` floats are K. The next `head_dim` floats are V.
pub fn write_token(
    pool: &mut BlockPool,
    loc: TokenLoc,
    head_dim: u32,
    token_id: u32,
) -> Result<(), AllocError> {
    let values = token_kv(pool, head_dim, token_id)?;
    pool.kv_mut(loc.block, loc.slot)?.copy_from_slice(&values);
    Ok(())
}

/// Dot `query_token`'s K with every K stored in `table`, in token order.
pub fn attend(
    pool: &BlockPool,
    table: &BlockTable,
    head_dim: u32,
    query_token: u32,
) -> Result<Vec<f32>, AllocError> {
    let query = token_kv(pool, head_dim, query_token)?;
    let dim = head_dim as usize;
    let mut scores = Vec::with_capacity(table.len() as usize);
    for index in 0..table.len() {
        let loc = table.token_loc(pool, index)?;
        let kv = pool.kv(loc.block, loc.slot)?;
        let mut dot = 0.0f32;
        for (q, k) in query.iter().zip(kv.iter()).take(dim) {
            dot += q * k;
        }
        scores.push(dot);
    }
    Ok(scores)
}

fn token_kv(pool: &BlockPool, head_dim: u32, token_id: u32) -> Result<Vec<f32>, AllocError> {
    let width = pool.floats_per_token().ok_or(AllocError::NoStorage)?;
    if head_dim == 0 || width != head_dim.saturating_mul(2) {
        return Err(AllocError::KvWidth {
            floats_per_token: width,
            head_dim,
        });
    }
    let dim = head_dim as usize;
    let mut out = vec![0.0; dim * 2];
    let scale = token_id.saturating_add(1) as f32;
    for i in 0..dim {
        let k = scale * ((i as u32).saturating_add(1) as f32);
        out[i] = k;
        out[dim + i] = k * 0.5;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlockPool;

    #[test]
    fn fill_then_attend_reads_the_stored_keys_and_free_zeroes_them() {
        let head_dim = 2;
        let mut pool = BlockPool::with_storage(4, 2, head_dim * 2);
        let mut table = crate::BlockTable::new();
        fill_tokens(&mut pool, &mut table, head_dim, &[1, 2, 3]).unwrap();

        assert_eq!(table.len(), 3);
        assert_eq!(table.blocks().len(), 2);
        let tail = table.token_loc(&pool, 2).unwrap();
        assert_eq!(
            pool.kv(tail.block, tail.slot).unwrap(),
            &[4.0, 8.0, 2.0, 4.0]
        );
        assert_eq!(
            attend(&pool, &table, head_dim, 3).unwrap(),
            vec![40.0, 60.0, 80.0]
        );

        let blocks = table.blocks().to_vec();
        table.release(&mut pool).unwrap();
        // Free pushes onto a stack, so the last block released is next.
        let reused = pool.allocate().unwrap();
        assert_eq!(reused, *blocks.last().unwrap());
        assert_eq!(pool.kv(reused, 0).unwrap(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(pool.kv(reused, 1).unwrap(), &[0.0, 0.0, 0.0, 0.0]);
        pool.free(reused).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn attend_rejects_a_width_that_is_not_two_halves() {
        let mut pool = BlockPool::with_storage(1, 2, 3);
        let mut table = crate::BlockTable::new();
        table.grow(&mut pool, 1).unwrap();
        let loc = table.token_loc(&pool, 0).unwrap();
        assert_eq!(
            write_token(&mut pool, loc, 2, 1).unwrap_err(),
            AllocError::KvWidth {
                floats_per_token: 3,
                head_dim: 2,
            }
        );
        table.release(&mut pool).unwrap();
    }
}
