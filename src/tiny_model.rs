//! One transformer block with fixed weights.
//!
//! The weights are a pure function of [`SEED`]. They live in this file, not
//! in a checkpoint, and they are not trained. The block is a single head of
//! attention, a ReLU feed-forward, and a tied embedding used as the output
//! projection. [`TinyModel::forward`] reads the K and V already stored for
//! the sequence and returns the next token id.

use crate::tokenizer::VOCAB_SIZE;
use crate::AllocError;

/// Seed for every weight in this block.
pub const SEED: u64 = 0xA11C_E001;

/// Next-token distribution and the mixed value vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelForward {
    /// Argmax over the vocabulary. Ties take the smaller id.
    pub next_token: u32,
    /// Softmax over the stored keys, in token order.
    pub scores: Vec<f32>,
    /// Weighted sum of the stored values. Length is the model dimension.
    pub context: Vec<f32>,
}

/// Fixed-weight block for one head dimension.
#[derive(Debug, Clone)]
pub struct TinyModel {
    dim: u32,
    embed: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
}

impl TinyModel {
    /// Build the block for `dim`.
    ///
    /// # Panics
    /// Panics if `dim` is zero.
    pub fn new(dim: u32) -> Self {
        assert!(dim > 0, "model dimension must be non-zero");
        let dim_usize = dim as usize;
        let vocab = VOCAB_SIZE as usize;
        let ff = dim_usize * 2;
        let mut embed = Vec::with_capacity(vocab * dim_usize);
        for token in 0..VOCAB_SIZE {
            for i in 0..dim {
                embed.push(weight(1, token * dim + i, 0.2));
            }
        }
        Self {
            dim,
            embed,
            wq: matrix(2, dim_usize, dim_usize),
            wk: matrix(3, dim_usize, dim_usize),
            wv: matrix(4, dim_usize, dim_usize),
            wo: matrix(5, dim_usize, dim_usize),
            w1: matrix(6, ff, dim_usize),
            w2: matrix(7, dim_usize, ff),
        }
    }

    /// Head dimension. Stored K and V are each this long.
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// K and V for `token`, concatenated. Length is `dim * 2`.
    pub fn kv(&self, token: u32) -> Result<Vec<f32>, AllocError> {
        let hidden = self.embed(token)?;
        let mut row = matvec(&self.wk, &hidden, self.dim as usize, self.dim as usize);
        row.extend(matvec(
            &self.wv,
            &hidden,
            self.dim as usize,
            self.dim as usize,
        ));
        Ok(row)
    }

    /// Run the block over `token_ids` and the stored rows.
    ///
    /// `rows[i]` is the K and V for `token_ids[i]`, length `dim * 2`. The
    /// query is the projection of the last token. An empty sequence uses a
    /// zero hidden state, so the next id does not depend on a prompt.
    pub fn forward(
        &self,
        token_ids: &[u32],
        rows: &[Vec<f32>],
    ) -> Result<ModelForward, AllocError> {
        let dim = self.dim as usize;
        if token_ids.len() != rows.len() {
            return Err(AllocError::TokenOutOfRange {
                index: token_ids.len() as u32,
                len: rows.len() as u32,
            });
        }
        if token_ids.is_empty() {
            let hidden = vec![0.0; dim];
            return Ok(ModelForward {
                next_token: self.argmax(&hidden),
                scores: Vec::new(),
                context: hidden,
            });
        }

        let hidden = self.embed(*token_ids.last().expect("sequence is not empty"))?;
        let query = matvec(&self.wq, &hidden, dim, dim);
        let scale = 1.0 / (self.dim as f32).sqrt();
        let mut logits = Vec::with_capacity(rows.len());
        let mut max_logit = f32::NEG_INFINITY;
        for row in rows {
            if row.len() != dim * 2 {
                return Err(AllocError::KvWidth {
                    floats_per_token: row.len() as u32,
                    head_dim: self.dim,
                });
            }
            let mut dot = 0.0f32;
            for (q, k) in query.iter().zip(&row[..dim]) {
                dot += q * k;
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
        for (row, weight) in rows.iter().zip(scores.iter().copied()) {
            for (slot, value) in context.iter_mut().zip(&row[dim..]) {
                *slot += weight * value;
            }
        }
        let mixed = matvec(&self.wo, &context, dim, dim);
        let mut hidden = hidden;
        for (slot, value) in hidden.iter_mut().zip(&mixed) {
            *slot += value;
        }
        let ff = dim * 2;
        let mut up = matvec(&self.w1, &hidden, ff, dim);
        for value in &mut up {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
        let down = matvec(&self.w2, &up, dim, ff);
        for (slot, value) in hidden.iter_mut().zip(&down) {
            *slot += value;
        }
        Ok(ModelForward {
            next_token: self.argmax(&hidden),
            scores,
            context,
        })
    }

    /// Bit mix of every weight. Tests pin this so the block cannot drift.
    pub fn checksum(&self) -> u64 {
        let mut acc = 0u64;
        for (index, value) in self
            .embed
            .iter()
            .chain(&self.wq)
            .chain(&self.wk)
            .chain(&self.wv)
            .chain(&self.wo)
            .chain(&self.w1)
            .chain(&self.w2)
            .copied()
            .enumerate()
        {
            acc ^= (value.to_bits() as u64).wrapping_mul(index as u64 + 1);
        }
        acc
    }

    fn embed(&self, token: u32) -> Result<Vec<f32>, AllocError> {
        if token >= VOCAB_SIZE {
            return Err(AllocError::TokenOutOfVocab {
                id: token,
                vocab: VOCAB_SIZE,
            });
        }
        let dim = self.dim as usize;
        let start = token as usize * dim;
        Ok(self.embed[start..start + dim].to_vec())
    }

    fn argmax(&self, hidden: &[f32]) -> u32 {
        let dim = self.dim as usize;
        let mut best_id = 0u32;
        let mut best = f32::NEG_INFINITY;
        for token in 0..VOCAB_SIZE {
            let start = token as usize * dim;
            let mut dot = 0.0f32;
            for (h, e) in hidden.iter().zip(&self.embed[start..start + dim]) {
                dot += h * e;
            }
            if dot > best {
                best = dot;
                best_id = token;
            }
        }
        best_id
    }
}

fn matrix(stream: u64, rows: usize, cols: usize) -> Vec<f32> {
    let scale = 1.0 / (cols as f32).sqrt();
    let mut out = Vec::with_capacity(rows * cols);
    for index in 0..(rows * cols) as u32 {
        out.push(weight(stream, index, scale));
    }
    out
}

fn matvec(matrix: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows];
    for (row, slot) in out.iter_mut().enumerate() {
        let start = row * cols;
        let mut sum = 0.0f32;
        for (weight, value) in matrix[start..start + cols].iter().zip(input) {
            sum += weight * value;
        }
        *slot = sum;
    }
    out
}

fn weight(stream: u64, index: u32, scale: f32) -> f32 {
    let mut x = SEED
        ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(index).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    let unit = (x as u32 as f32) / (u32::MAX as f32);
    (unit * 2.0 - 1.0) * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_do_not_drift() {
        assert_eq!(TinyModel::new(4).checksum(), 485_211_148_041);
    }

    #[test]
    fn forward_reads_the_stored_rows() {
        let model = TinyModel::new(2);
        let tokens = [1u32, 2, 3];
        let rows: Vec<Vec<f32>> = tokens
            .iter()
            .copied()
            .map(|token| model.kv(token).unwrap())
            .collect();
        let first = model.forward(&tokens, &rows).unwrap();
        assert_eq!(first.scores.len(), 3);
        assert_eq!(first.context.len(), 2);
        let sum: f32 = first.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(first.next_token < VOCAB_SIZE);

        let mut changed = rows.clone();
        for value in &mut changed[0][..2] {
            *value = 3.0;
        }
        let second = model.forward(&tokens, &changed).unwrap();
        assert_ne!(first.scores, second.scores);
    }
}
