//! Python wrappers around the Rust pool, tables, prefix cache, and scheduler.
//!
//! The core crate stays free of PyO3. This module is the optional binding.
//! `BlockPool.with_storage` keeps an f32 vector per token. `release` and
//! `clear` return block references to the pool; dropping a Python object
//! does not.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use ::paged_kv::{AllocError, Phase, PreemptError, PreemptPolicy};

#[pyclass]
struct BlockPool {
    inner: ::paged_kv::BlockPool,
}

#[pymethods]
impl BlockPool {
    #[new]
    fn new(num_blocks: u32, block_size: u32) -> Self {
        Self {
            inner: ::paged_kv::BlockPool::new(num_blocks, block_size),
        }
    }

    fn block_size(&self) -> u32 {
        self.inner.block_size()
    }

    fn capacity(&self) -> u32 {
        self.inner.capacity()
    }

    fn free_count(&self) -> u32 {
        self.inner.free_count()
    }

    /// Pool that stores `floats_per_token` f32 values in every token slot.
    #[staticmethod]
    fn with_storage(num_blocks: u32, block_size: u32, floats_per_token: u32) -> Self {
        Self {
            inner: ::paged_kv::BlockPool::with_storage(num_blocks, block_size, floats_per_token),
        }
    }

    fn floats_per_token(&self) -> Option<u32> {
        self.inner.floats_per_token()
    }

    fn kv(&self, block: u32, slot: u32) -> PyResult<Vec<f32>> {
        self.inner
            .kv(::paged_kv::BlockId::from_index(block), slot)
            .map(|values| values.to_vec())
            .map_err(alloc_err)
    }

    fn set_kv(&mut self, block: u32, slot: u32, values: Vec<f32>) -> PyResult<()> {
        let dst = self
            .inner
            .kv_mut(::paged_kv::BlockId::from_index(block), slot)
            .map_err(alloc_err)?;
        if dst.len() != values.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} floats, got {}",
                dst.len(),
                values.len()
            )));
        }
        dst.copy_from_slice(&values);
        Ok(())
    }
}

#[pyclass]
struct BlockTable {
    inner: ::paged_kv::BlockTable,
}

#[pymethods]
impl BlockTable {
    #[new]
    fn new() -> Self {
        Self {
            inner: ::paged_kv::BlockTable::new(),
        }
    }

    #[getter]
    fn len(&self) -> u32 {
        self.inner.len()
    }

    fn grow(&mut self, pool: &mut BlockPool, new_len: u32) -> PyResult<()> {
        self.inner.grow(&mut pool.inner, new_len).map_err(alloc_err)
    }

    fn fork_prefix(&self, pool: &mut BlockPool, prefix_len: u32) -> PyResult<Self> {
        self.inner
            .fork_prefix(&mut pool.inner, prefix_len)
            .map(|inner| Self { inner })
            .map_err(alloc_err)
    }

    /// `(block_index, slot)` for logical token `index`.
    fn token_loc(&self, pool: &BlockPool, index: u32) -> PyResult<(u32, u32)> {
        let loc = self
            .inner
            .token_loc(&pool.inner, index)
            .map_err(alloc_err)?;
        Ok((loc.block.index(), loc.slot))
    }

    /// Drop this table's block references. A second call releases nothing.
    fn release(&mut self, pool: &mut BlockPool) -> PyResult<()> {
        let table = std::mem::take(&mut self.inner);
        table.release(&mut pool.inner).map_err(alloc_err)
    }
}

#[pyclass]
struct ContiguousTable {
    inner: ::paged_kv::ContiguousTable,
}

#[pymethods]
impl ContiguousTable {
    #[new]
    fn new() -> Self {
        Self {
            inner: ::paged_kv::ContiguousTable::new(),
        }
    }

    #[getter]
    fn len(&self) -> u32 {
        self.inner.len()
    }

    #[getter]
    fn capacity(&self) -> u32 {
        self.inner.capacity()
    }

    fn reserve_exact(&mut self, tokens: u32) {
        self.inner.reserve_exact(tokens);
    }

    fn grow(&mut self, new_len: u32) -> PyResult<()> {
        self.inner.grow(new_len).map_err(alloc_err)
    }
}

#[pyclass]
struct PrefixCache {
    inner: ::paged_kv::PrefixCache,
}

#[pymethods]
impl PrefixCache {
    #[new]
    fn new() -> Self {
        Self {
            inner: ::paged_kv::PrefixCache::new(),
        }
    }

    fn hits(&self) -> u64 {
        self.inner.hits()
    }

    fn misses(&self) -> u64 {
        self.inner.misses()
    }

    fn hit_rate(&self) -> f64 {
        self.inner.hit_rate()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn insert(&mut self, pool: &mut BlockPool, key: u64, source: &BlockTable) -> PyResult<()> {
        self.inner
            .insert(&mut pool.inner, key, &source.inner)
            .map_err(alloc_err)
    }

    fn lookup(
        &mut self,
        pool: &mut BlockPool,
        key: u64,
        prefix_len: u32,
    ) -> PyResult<Option<BlockTable>> {
        self.inner
            .lookup(&mut pool.inner, key, prefix_len)
            .map(|found| found.map(|inner| BlockTable { inner }))
            .map_err(alloc_err)
    }

    fn clear(&mut self, pool: &mut BlockPool) -> PyResult<()> {
        self.inner.clear(&mut pool.inner).map_err(alloc_err)
    }
}

#[pyclass]
struct Scheduler {
    inner: ::paged_kv::Scheduler,
}

#[pymethods]
impl Scheduler {
    /// Build a scheduler and the pool it owns.
    #[new]
    fn new(num_blocks: u32, block_size: u32, max_batch: usize) -> Self {
        Self {
            inner: ::paged_kv::Scheduler::new(
                ::paged_kv::BlockPool::new(num_blocks, block_size),
                max_batch,
            ),
        }
    }

    fn add(&mut self, prompt_len: u32, max_new_tokens: u32) -> u64 {
        self.inner.add(prompt_len, max_new_tokens).get()
    }

    fn admit(&mut self) -> PyResult<Vec<u64>> {
        self.inner
            .admit()
            .map(|ids| ids.into_iter().map(|id| id.get()).collect())
            .map_err(alloc_err)
    }

    /// Returns `(decoded_ids, finished_ids)`.
    fn step(&mut self) -> PyResult<(Vec<u64>, Vec<u64>)> {
        let step = self.inner.step().map_err(alloc_err)?;
        Ok((
            step.decoded.into_iter().map(|id| id.get()).collect(),
            step.finished.into_iter().map(|id| id.get()).collect(),
        ))
    }

    fn retire(&mut self) -> PyResult<Vec<u64>> {
        self.inner
            .retire()
            .map(|ids| ids.into_iter().map(|id| id.get()).collect())
            .map_err(alloc_err)
    }

    /// `policy` is `"pause"` or `"recompute"`.
    fn preempt(&mut self, policy: &str) -> PyResult<u64> {
        let policy = match policy {
            "pause" => PreemptPolicy::Pause,
            "recompute" => PreemptPolicy::Recompute,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown preempt policy '{other}', expected 'pause' or 'recompute'"
                )));
            }
        };
        self.inner
            .preempt(policy)
            .map(|id| id.get())
            .map_err(preempt_err)
    }

    fn waiting_len(&self) -> usize {
        self.inner.waiting_len()
    }

    fn is_idle(&self) -> bool {
        self.inner.is_idle()
    }

    fn free_count(&self) -> u32 {
        self.inner.pool().free_count()
    }

    /// Rows of `(id, phase, length, generated)` for the batch, oldest first.
    fn running(&self) -> Vec<(u64, String, u32, u32)> {
        self.inner
            .running()
            .iter()
            .map(|seq| {
                (
                    seq.id().get(),
                    phase_name(seq.phase()).to_string(),
                    seq.len(),
                    seq.generated(),
                )
            })
            .collect()
    }
}

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Waiting => "waiting",
        Phase::Decoding => "decoding",
        Phase::Finished => "finished",
        Phase::Paused => "paused",
    }
}

fn alloc_err(err: AllocError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn preempt_err(err: PreemptError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Allocate blocks for `token_ids` and write each token's K and V.
#[pyfunction]
fn fill_tokens(
    pool: &mut BlockPool,
    table: &mut BlockTable,
    head_dim: u32,
    token_ids: Vec<u32>,
) -> PyResult<()> {
    ::paged_kv::fill_tokens(&mut pool.inner, &mut table.inner, head_dim, &token_ids)
        .map_err(alloc_err)
}

/// Softmax weights over stored K, and the weighted sum of stored V.
///
/// Returns `(scores, context)`. The query is the last token's stored K.
#[pyfunction]
fn attend(pool: &BlockPool, table: &BlockTable, head_dim: u32) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let step = ::paged_kv::attend(&pool.inner, &table.inner, head_dim).map_err(alloc_err)?;
    Ok((step.scores, step.context))
}

/// Byte ids for `text`. Token `b` is the raw byte `b`.
#[pyfunction]
fn encode(text: &str) -> Vec<u32> {
    ::paged_kv::encode(text)
}

/// Text for `ids`. Ids outside a byte are skipped.
#[pyfunction]
fn decode(ids: Vec<u32>) -> String {
    ::paged_kv::decode(&ids)
}

/// Run the fixed block on `text`.
///
/// Returns `(ids, text)`. `ids` starts with the byte ids of `text`, then up
/// to `max_new_tokens` ids the block chose. `text` is those ids decoded.
#[pyfunction]
fn generate(text: &str, max_new_tokens: u32) -> PyResult<(Vec<u32>, String)> {
    let generated = ::paged_kv::generate(text, max_new_tokens).map_err(alloc_err)?;
    Ok((generated.ids, generated.text))
}

/// Paged KV blocks, stored f32 vectors, and an FCFS batcher.
#[pymodule]
fn paged_kv(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BlockPool>()?;
    module.add_class::<BlockTable>()?;
    module.add_class::<ContiguousTable>()?;
    module.add_class::<PrefixCache>()?;
    module.add_class::<Scheduler>()?;
    module.add_function(wrap_pyfunction!(fill_tokens, module)?)?;
    module.add_function(wrap_pyfunction!(attend, module)?)?;
    module.add_function(wrap_pyfunction!(encode, module)?)?;
    module.add_function(wrap_pyfunction!(decode, module)?)?;
    module.add_function(wrap_pyfunction!(generate, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_python_caller_reserves_shares_and_preempts() {
        Python::attach(|py| {
            let module = PyModule::new(py, "paged_kv").unwrap();
            paged_kv(&module).unwrap();
            py.import("sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item("paged_kv", &module)
                .unwrap();

            py.run(
                c"
import paged_kv

pool = paged_kv.BlockPool(8, 4)
table = paged_kv.BlockTable()
table.grow(pool, 6)
assert table.len == 6
child = table.fork_prefix(pool, 4)
assert child.len == 4

cache = paged_kv.PrefixCache()
assert cache.lookup(pool, 1, 4) is None
cache.insert(pool, 1, table)
hit = cache.lookup(pool, 1, 4)
assert hit is not None and hit.len == 4
assert cache.hits() == 1 and cache.misses() == 1
assert cache.hit_rate() == 0.5
hit.release(pool)
child.release(pool)
table.release(pool)
cache.clear(pool)
assert pool.free_count() == pool.capacity()

span = paged_kv.ContiguousTable()
span.reserve_exact(10)
span.grow(10)
assert span.len == 10 and span.capacity == 10

sched = paged_kv.Scheduler(8, 4, 1)
first = sched.add(4, 3)
second = sched.add(4, 1)
assert sched.admit() == [first]
decoded, finished = sched.step()
assert decoded == [first] and finished == []
assert sched.preempt('pause') == first
assert sched.running() == []
assert sched.admit() == [second]
decoded, finished = sched.step()
assert finished == [second]
assert sched.retire() == [second]
assert sched.admit() == [first]
row = sched.running()[0]
assert row[0] == first and row[1] == 'decoding' and row[3] == 1
assert sched.preempt('recompute') == first
assert sched.running() == []
assert sched.waiting_len() == 1
assert sched.free_count() == 8
assert sched.is_idle() is False

stored = paged_kv.BlockPool.with_storage(4, 2, 4)
seq = paged_kv.BlockTable()
paged_kv.fill_tokens(stored, seq, 2, [1, 2, 3])
assert seq.len == 3
scores, context = paged_kv.attend(stored, seq, 2)
assert len(scores) == 3 and abs(sum(scores) - 1.0) < 1e-5
assert scores[2] > scores[1] > scores[0]
assert len(context) == 2
block, slot = seq.token_loc(stored, 2)
assert stored.kv(block, slot) == [4.0, 8.0, 2.0, 4.0]
stored.set_kv(block, slot, [9.0, 9.0, 9.0, 9.0])
assert stored.kv(block, slot) == [9.0, 9.0, 9.0, 9.0]
seq.release(stored)
assert stored.free_count() == stored.capacity()

assert paged_kv.encode('hi') == [104, 105]
assert paged_kv.decode([104, 256, 105]) == 'hi'
ids, text = paged_kv.generate('hi', 2)
assert ids == [104, 105, 34, 7]
assert text == paged_kv.decode(ids)
assert ids[2:] != [106, 107]
ids, text = paged_kv.generate('hi', 0)
assert ids == [104, 105] and text == 'hi'
",
                None,
                None,
            )
            .inspect_err(|err| err.print(py))
            .expect("python caller script");
        });
    }
}
