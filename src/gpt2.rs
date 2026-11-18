//! One loaded model: `openai-community/gpt2`.
//!
//! The loader is the `gpt2` feature. The first call downloads
//! `model.safetensors` and `tokenizer.json` into the Hugging Face cache.
//! Later calls reuse those files. Without the feature nothing is downloaded.
//! The forward writes one K and one V vector per layer into a flat row the
//! block pool can store, and the next id is the argmax of this model's logits.
//!
//! This is greedy and single-token. There is no batch of hidden states and
//! no server.

#[cfg(feature = "gpt2")]
use std::collections::HashMap;
#[cfg(feature = "gpt2")]
use std::sync::OnceLock;

#[cfg(feature = "gpt2")]
use candle_core::{Device, Tensor};
#[cfg(feature = "gpt2")]
use candle_nn::ops::layer_norm;
#[cfg(feature = "gpt2")]
use tokenizers::Tokenizer;

#[cfg(feature = "gpt2")]
use crate::AllocError;

/// Hub repo the `gpt2` feature downloads.
pub const MODEL_ID: &str = "openai-community/gpt2";
/// GPT-2 small.
pub const N_LAYER: usize = 12;
/// Hidden size.
pub const N_EMBD: usize = 768;
/// Attention heads.
pub const N_HEAD: usize = 12;
/// Width of one head.
pub const HEAD_DIM_USIZE: usize = N_EMBD / N_HEAD;
/// [`HEAD_DIM_USIZE`] as `u32`.
pub const HEAD_DIM: u32 = HEAD_DIM_USIZE as u32;
/// Context the position table covers.
#[cfg_attr(not(feature = "gpt2"), allow(dead_code))]
pub const N_POSITIONS: usize = 1024;
/// Byte-pair vocabulary, including the end-of-text id.
pub const MODEL_VOCAB: u32 = 50_257;
/// `<|endoftext|>`.
pub const MODEL_EOS: u32 = 50_256;
/// Floats stored per token: every layer's K and V, each [`N_EMBD`] wide.
pub const KV_FLOATS: u32 = (N_LAYER * 2 * N_EMBD) as u32;

#[cfg(feature = "gpt2")]
const LN_EPS: f32 = 1e-5;
#[cfg(feature = "gpt2")]
const N_INNER: usize = N_EMBD * 4;

/// Result of running one token through every layer.
#[cfg(feature = "gpt2")]
#[derive(Debug, Clone)]
pub struct TokenStep {
    /// K and V for this token, layer 0 first. Length is [`KV_FLOATS`].
    pub kv: Vec<f32>,
    /// Argmax of the tied output projection. Ties take the smaller id.
    pub next_token: u32,
    /// Mean of the last layer's head scores. One weight per token so far.
    pub scores: Vec<f32>,
    /// Last layer's merged attention output, before the output projection.
    pub context: Vec<f32>,
}

#[cfg(feature = "gpt2")]
struct Layer {
    ln1_w: Tensor,
    ln1_b: Tensor,
    attn_w: Tensor,
    attn_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    fc_w: Tensor,
    fc_b: Tensor,
    mlp_w: Tensor,
    mlp_b: Tensor,
}

/// Weights and tokenizer for [`MODEL_ID`].
#[cfg(feature = "gpt2")]
pub struct Gpt2 {
    wte: Tensor,
    wpe: Tensor,
    layers: Vec<Layer>,
    ln_f_w: Tensor,
    ln_f_b: Tensor,
    tokenizer: Tokenizer,
}

#[cfg(feature = "gpt2")]
impl Gpt2 {
    /// Download the weights if needed, then map them.
    pub fn load() -> Result<Self, AllocError> {
        let device = Device::Cpu;
        let client =
            hf_hub::HFClientSync::new().map_err(|err| AllocError::Model(err.to_string()))?;
        let repo = client.model("openai-community", "gpt2");
        let weights_path = repo
            .download_file()
            .filename("model.safetensors")
            .send()
            .map_err(|err| AllocError::Model(err.to_string()))?;
        let tokenizer_path = repo
            .download_file()
            .filename("tokenizer.json")
            .send()
            .map_err(|err| AllocError::Model(err.to_string()))?;
        let weights = candle_core::safetensors::load(weights_path, &device)
            .map_err(|err| AllocError::Model(err.to_string()))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|err| AllocError::Model(err.to_string()))?;
        Self::from_weights(weights, tokenizer)
    }

    fn from_weights(
        mut weights: HashMap<String, Tensor>,
        tokenizer: Tokenizer,
    ) -> Result<Self, AllocError> {
        let wte = take(&mut weights, "wte.weight")?;
        let wpe = take(&mut weights, "wpe.weight")?;
        expect_shape(&wte, &[MODEL_VOCAB as usize, N_EMBD], "wte.weight")?;
        expect_shape(&wpe, &[N_POSITIONS, N_EMBD], "wpe.weight")?;
        let mut layers = Vec::with_capacity(N_LAYER);
        for index in 0..N_LAYER {
            let prefix = format!("h.{index}");
            layers.push(Layer {
                ln1_w: take(&mut weights, &format!("{prefix}.ln_1.weight"))?,
                ln1_b: take(&mut weights, &format!("{prefix}.ln_1.bias"))?,
                attn_w: expect_in_out(
                    take(&mut weights, &format!("{prefix}.attn.c_attn.weight"))?,
                    N_EMBD,
                    3 * N_EMBD,
                    "c_attn.weight",
                )?,
                attn_b: take(&mut weights, &format!("{prefix}.attn.c_attn.bias"))?,
                proj_w: expect_in_out(
                    take(&mut weights, &format!("{prefix}.attn.c_proj.weight"))?,
                    N_EMBD,
                    N_EMBD,
                    "c_proj.weight",
                )?,
                proj_b: take(&mut weights, &format!("{prefix}.attn.c_proj.bias"))?,
                ln2_w: take(&mut weights, &format!("{prefix}.ln_2.weight"))?,
                ln2_b: take(&mut weights, &format!("{prefix}.ln_2.bias"))?,
                fc_w: expect_in_out(
                    take(&mut weights, &format!("{prefix}.mlp.c_fc.weight"))?,
                    N_EMBD,
                    N_INNER,
                    "c_fc.weight",
                )?,
                fc_b: take(&mut weights, &format!("{prefix}.mlp.c_fc.bias"))?,
                mlp_w: expect_in_out(
                    take(&mut weights, &format!("{prefix}.mlp.c_proj.weight"))?,
                    N_INNER,
                    N_EMBD,
                    "mlp.c_proj.weight",
                )?,
                mlp_b: take(&mut weights, &format!("{prefix}.mlp.c_proj.bias"))?,
            });
        }
        let ln_f_w = take(&mut weights, "ln_f.weight")?;
        let ln_f_b = take(&mut weights, "ln_f.bias")?;
        Ok(Self {
            wte,
            wpe,
            layers,
            ln_f_w,
            ln_f_b,
            tokenizer,
        })
    }

    /// Token ids for `text`, with no extra special token added.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, AllocError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| AllocError::Model(err.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Text for `ids`. The end-of-text id is skipped.
    pub fn decode(&self, ids: &[u32]) -> Result<String, AllocError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|err| AllocError::Model(err.to_string()))
    }

    /// Run `token` at `position`, attending over `past` and this token.
    ///
    /// `past` is every earlier token's [`TokenStep::kv`], in order. The
    /// returned row is what the pool stores for this token.
    pub fn forward_token(
        &self,
        token: u32,
        position: usize,
        past: &[Vec<f32>],
    ) -> Result<TokenStep, AllocError> {
        if token >= MODEL_VOCAB {
            return Err(AllocError::TokenOutOfVocab {
                id: token,
                vocab: MODEL_VOCAB,
            });
        }
        if position >= N_POSITIONS {
            return Err(AllocError::Model(format!(
                "position {position} is past {MODEL_ID}'s context of {N_POSITIONS}"
            )));
        }
        for row in past {
            if row.len() != KV_FLOATS as usize {
                return Err(AllocError::KvWidth {
                    floats_per_token: row.len() as u32,
                    head_dim: HEAD_DIM,
                });
            }
        }
        if past.len() != position {
            return Err(AllocError::Model(format!(
                "expected {position} stored rows before this token, found {}",
                past.len()
            )));
        }

        let token_vec = row(&self.wte, token as usize)?;
        let pos_vec = row(&self.wpe, position)?;
        let mut hidden = token_vec.broadcast_add(&pos_vec).map_err(model_err)?;
        let mut kv = Vec::with_capacity(KV_FLOATS as usize);
        let mut scores = Vec::new();
        let mut context = Vec::new();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let normed =
                layer_norm(&hidden, &layer.ln1_w, &layer.ln1_b, LN_EPS).map_err(model_err)?;
            let qkv = linear(&normed, &layer.attn_w, &layer.attn_b)?;
            let qkv = qkv.to_vec1::<f32>().map_err(model_err)?;
            if qkv.len() != 3 * N_EMBD {
                return Err(AllocError::Model(format!(
                    "c_attn produced {} floats",
                    qkv.len()
                )));
            }
            let (q, rest) = qkv.split_at(N_EMBD);
            let (k, v) = rest.split_at(N_EMBD);
            kv.extend_from_slice(k);
            kv.extend_from_slice(v);

            let mut keys: Vec<&[f32]> = Vec::with_capacity(position + 1);
            let mut values: Vec<&[f32]> = Vec::with_capacity(position + 1);
            let offset = layer_index * 2 * N_EMBD;
            for earlier in past {
                keys.push(&earlier[offset..offset + N_EMBD]);
                values.push(&earlier[offset + N_EMBD..offset + 2 * N_EMBD]);
            }
            keys.push(k);
            values.push(v);
            let attended = attend(q, &keys, &values);
            if layer_index + 1 == self.layers.len() {
                scores = attended.scores;
                context = attended.context.clone();
            }
            let merged =
                Tensor::from_vec(attended.context, N_EMBD, hidden.device()).map_err(model_err)?;
            let projected = linear(&merged, &layer.proj_w, &layer.proj_b)?;
            hidden = hidden.broadcast_add(&projected).map_err(model_err)?;

            let normed =
                layer_norm(&hidden, &layer.ln2_w, &layer.ln2_b, LN_EPS).map_err(model_err)?;
            let mid = linear(&normed, &layer.fc_w, &layer.fc_b)?;
            let mid = mid.gelu().map_err(model_err)?;
            let mlp = linear(&mid, &layer.mlp_w, &layer.mlp_b)?;
            hidden = hidden.broadcast_add(&mlp).map_err(model_err)?;
        }
        let normed = layer_norm(&hidden, &self.ln_f_w, &self.ln_f_b, LN_EPS).map_err(model_err)?;
        let logits = linear_no_bias(&normed, &self.wte.t().map_err(model_err)?)?;
        let logits = logits.to_vec1::<f32>().map_err(model_err)?;
        let next_token = argmax(&logits);
        Ok(TokenStep {
            kv,
            next_token,
            scores,
            context,
        })
    }
}

#[cfg(feature = "gpt2")]
struct Attended {
    context: Vec<f32>,
    scores: Vec<f32>,
}

#[cfg(feature = "gpt2")]
fn attend(q: &[f32], keys: &[&[f32]], values: &[&[f32]]) -> Attended {
    let steps = keys.len();
    let scale = 1.0 / (HEAD_DIM_USIZE as f32).sqrt();
    let mut mean = vec![0.0f32; steps];
    let mut merged = vec![0.0f32; N_EMBD];
    for head in 0..N_HEAD {
        let q_at = head * HEAD_DIM_USIZE;
        let mut scores = vec![0.0f32; steps];
        let mut max_score = f32::NEG_INFINITY;
        for (step, key) in keys.iter().enumerate() {
            let mut dot = 0.0f32;
            let k_at = head * HEAD_DIM_USIZE;
            for dim in 0..HEAD_DIM_USIZE {
                dot += q[q_at + dim] * key[k_at + dim];
            }
            let score = dot * scale;
            scores[step] = score;
            if score > max_score {
                max_score = score;
            }
        }
        let mut sum = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            sum += *score;
        }
        let inv = 1.0 / sum;
        for (step, score) in scores.iter_mut().enumerate() {
            *score *= inv;
            mean[step] += *score;
        }
        for (step, value) in values.iter().enumerate() {
            let weight = scores[step];
            let v_at = head * HEAD_DIM_USIZE;
            let out_at = head * HEAD_DIM_USIZE;
            for dim in 0..HEAD_DIM_USIZE {
                merged[out_at + dim] += weight * value[v_at + dim];
            }
        }
    }
    let heads = N_HEAD as f32;
    for score in &mut mean {
        *score /= heads;
    }
    Attended {
        context: merged,
        scores: mean,
    }
}

#[cfg(feature = "gpt2")]
fn argmax(logits: &[f32]) -> u32 {
    let mut best_at = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (index, logit) in logits.iter().copied().enumerate() {
        if logit > best {
            best = logit;
            best_at = index as u32;
        }
    }
    best_at
}

#[cfg(feature = "gpt2")]
fn row(table: &Tensor, index: usize) -> Result<Tensor, AllocError> {
    table
        .narrow(0, index, 1)
        .and_then(|row| row.squeeze(0))
        .map_err(model_err)
}

#[cfg(feature = "gpt2")]
fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor, AllocError> {
    let y = x
        .unsqueeze(0)
        .and_then(|x| x.matmul(weight))
        .and_then(|y| y.squeeze(0))
        .and_then(|y| y.broadcast_add(bias))
        .map_err(model_err)?;
    Ok(y)
}

#[cfg(feature = "gpt2")]
fn linear_no_bias(x: &Tensor, weight: &Tensor) -> Result<Tensor, AllocError> {
    x.unsqueeze(0)
        .and_then(|x| x.matmul(weight))
        .and_then(|y| y.squeeze(0))
        .map_err(model_err)
}

#[cfg(feature = "gpt2")]
fn model_err(err: candle_core::Error) -> AllocError {
    AllocError::Model(err.to_string())
}

#[cfg(feature = "gpt2")]
fn take(weights: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor, AllocError> {
    if let Some(weight) = weights.remove(name) {
        return Ok(weight);
    }
    let prefixed = format!("transformer.{name}");
    weights
        .remove(&prefixed)
        .ok_or_else(|| AllocError::Model(format!("missing weight {name} in {MODEL_ID}")))
}

#[cfg(feature = "gpt2")]
fn expect_shape(weight: &Tensor, shape: &[usize], name: &str) -> Result<(), AllocError> {
    if weight.dims() == shape {
        Ok(())
    } else {
        Err(AllocError::Model(format!(
            "{name} has shape {:?}, expected {shape:?}",
            weight.dims()
        )))
    }
}

#[cfg(feature = "gpt2")]
fn expect_in_out(
    weight: Tensor,
    input: usize,
    output: usize,
    name: &str,
) -> Result<Tensor, AllocError> {
    if weight.dims() == [input, output] {
        return Ok(weight);
    }
    if weight.dims() == [output, input] {
        return weight
            .t()
            .and_then(|weight| weight.contiguous())
            .map_err(model_err);
    }
    Err(AllocError::Model(format!(
        "{name} has shape {:?}, expected {input} by {output}",
        weight.dims()
    )))
}

/// The process-wide GPT-2 weights. The first call downloads them.
#[cfg(feature = "gpt2")]
pub fn model() -> Result<&'static Gpt2, AllocError> {
    static MODEL: OnceLock<Gpt2> = OnceLock::new();
    if let Some(loaded) = MODEL.get() {
        return Ok(loaded);
    }
    let loaded = Gpt2::load()?;
    let _ = MODEL.set(loaded);
    MODEL
        .get()
        .ok_or_else(|| AllocError::Model(format!("{MODEL_ID} was loaded but could not be stored")))
}

/// Encode `text` with GPT-2's tokenizer.
#[cfg(feature = "gpt2")]
pub fn model_encode(text: &str) -> Result<Vec<u32>, AllocError> {
    model()?.encode(text)
}

/// Decode GPT-2 token ids. The end-of-text id is skipped.
#[cfg(feature = "gpt2")]
pub fn model_decode(ids: &[u32]) -> Result<String, AllocError> {
    model()?.decode(ids)
}

#[cfg(all(test, feature = "gpt2"))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "downloads openai-community/gpt2"]
    fn gpt2_next_token_comes_from_the_logits() {
        let model = Gpt2::load().expect("gpt2 weights");
        let ids = model.encode("The capital of France is").expect("encode");
        assert!(!ids.is_empty());
        let mut rows = Vec::new();
        let mut last = None;
        for (position, id) in ids.iter().copied().enumerate() {
            let step = model.forward_token(id, position, &rows).expect("forward");
            assert_eq!(step.kv.len(), KV_FLOATS as usize);
            assert_eq!(step.scores.len(), position + 1);
            let sum: f32 = step.scores.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "scores sum to {sum}");
            assert!(step.kv.iter().any(|value| *value != 0.0));
            rows.push(step.kv.clone());
            last = Some(step);
        }
        let step = last.expect("the prompt has a token");
        let text = model.decode(&[step.next_token]).expect("decode");
        assert_eq!(ids, vec![464, 3139, 286, 4881, 318]);
        // Greedy GPT-2 small picks " the". " Paris" ranks next to it, not first.
        assert_eq!(step.next_token, 262);
        assert_eq!(text, " the");
    }
}
