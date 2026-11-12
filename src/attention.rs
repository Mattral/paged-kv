//! One attention step over KV vectors stored in a [`BlockPool`](crate::BlockPool).
//!
//! [`fill_tokens`] writes K and V from a fixed function of the token id.
//! There is no learned weight and no tokenizer. [`attend`] does not use that
//! function. It reads the stored vectors: the query is the last token's K,
//! the scores are softmax over dots with every stored K, and the context is
//! those weights applied to the stored V.

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

/// One attention step over the K and V stored in `table`.
///
/// The query is the last token's stored K. `scores` are softmax weights in
/// token order and sum to 1. `context` is those weights applied to the
/// stored V vectors. An empty table returns no scores and a zero context.
/// Overwriting a stored vector changes the result. The token id is not read
/// here.
pub fn attend(
    pool: &BlockPool,
    table: &BlockTable,
    head_dim: u32,
) -> Result<Attention, AllocError> {
    let width = pool.floats_per_token().ok_or(AllocError::NoStorage)?;
    if head_dim == 0 || width != head_dim.saturating_mul(2) {
        return Err(AllocError::KvWidth {
            floats_per_token: width,
            head_dim,
        });
    }
    let dim = head_dim as usize;
    if table.is_empty() {
        return Ok(Attention {
            scores: Vec::new(),
            context: vec![0.0; dim],
        });
    }

    let last = table.token_loc(pool, table.len() - 1)?;
    let query = pool.kv(last.block, last.slot)?.to_vec();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut logits = Vec::with_capacity(table.len() as usize);
    let mut max_logit = f32::NEG_INFINITY;
    for index in 0..table.len() {
        let loc = table.token_loc(pool, index)?;
        let kv = pool.kv(loc.block, loc.slot)?;
        let mut dot = 0.0f32;
        for i in 0..dim {
            dot += query[i] * kv[i];
        }
        let logit = dot * scale;
        max_logit = max_logit.max(logit);
        logits.push(logit);
    }

    let mut scores = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for logit in &logits {
        let weight = (logit - max_logit).exp();
        scores.push(weight);
        sum += weight;
    }
    for weight in &mut scores {
        *weight /= sum;
    }

    let mut context = vec![0.0; dim];
    for (index, weight) in scores.iter().copied().enumerate() {
        let loc = table.token_loc(pool, index as u32)?;
        let kv = pool.kv(loc.block, loc.slot)?;
        for i in 0..dim {
            context[i] += weight * kv[dim + i];
        }
    }
    Ok(Attention { scores, context })
}

/// Softmax weights and the mixed value vector from [`attend`].
#[derive(Debug, Clone, PartialEq)]
pub struct Attention {
    /// Weight of each token, in order. Empty when the table is empty.
    pub scores: Vec<f32>,
    /// Weighted sum of stored V. Length is the head dimension.
    pub context: Vec<f32>,
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
        let step = attend(&pool, &table, head_dim).unwrap();
        assert_eq!(step.scores.len(), 3);
        assert_eq!(step.context.len(), head_dim as usize);
        let weight_sum: f32 = step.scores.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-5);
        assert!(step.scores[2] > step.scores[1] && step.scores[1] > step.scores[0]);

        let first = table.token_loc(&pool, 0).unwrap();
        {
            let row = pool.kv_mut(first.block, first.slot).unwrap();
            row[0] = 50.0;
            row[1] = 50.0;
        }
        let raised = attend(&pool, &table, head_dim).unwrap();
        assert!(raised.scores[0] > step.scores[0]);
        assert!(raised.scores[0] > raised.scores[2]);

        {
            let row = pool.kv_mut(first.block, first.slot).unwrap();
            row[0] = 2.0;
            row[1] = 4.0;
            row[2] = 1.0;
            row[3] = 2.0;
        }
        let last = table.token_loc(&pool, 2).unwrap();
        {
            let row = pool.kv_mut(last.block, last.slot).unwrap();
            row[2] = 10.0;
            row[3] = 10.0;
        }
        let valued = attend(&pool, &table, head_dim).unwrap();
        for (before, after) in step.scores.iter().zip(&valued.scores) {
            assert!((before - after).abs() < 1e-5);
        }
        assert_ne!(step.context, valued.context);

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
