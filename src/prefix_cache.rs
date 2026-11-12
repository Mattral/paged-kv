//! Caller-keyed cache of shared prompt prefixes.
//!
//! The cache does not see token bytes. The caller hashes a prompt, or a
//! prefix of one, and uses that `u64` as the key. A hit forks the cached
//! blocks with [`crate::BlockTable::fork_prefix`]. A miss only increments
//! the miss count.
//!
//! [`PrefixCache::hit_rate`] is `hits / (hits + misses)`. Lookups that fail
//! with [`crate::AllocError`] are neither. Inserting a table is not a lookup.

use std::collections::HashMap;

use crate::{AllocError, BlockPool, BlockTable};

/// Shared prefixes and how often lookup found one.
///
/// The cache holds its own block references. [`PrefixCache::clear`] drops
/// them. Dropping the cache without clearing leaks those references until
/// the pool is dropped.
#[derive(Debug)]
#[must_use = "call clear to return cached block references to the pool"]
pub struct PrefixCache {
    entries: HashMap<u64, BlockTable>,
    hits: u64,
    misses: u64,
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixCache {
    /// Empty cache. No blocks are held.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Successful lookups since this cache was created.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Lookups that found nothing long enough to fork.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Share of lookups that forked a cached prefix.
    ///
    /// Zero when nothing has been looked up. Allocation failures are not
    /// counted in either side of the ratio.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Cached keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no prefix is stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Store a fork of `source` under `key`.
    ///
    /// The cached fork covers the whole source table, so the caller can
    /// release `source` and the cache still holds the blocks. Replacing a
    /// key releases the previous fork.
    pub fn insert(
        &mut self,
        pool: &mut BlockPool,
        key: u64,
        source: &BlockTable,
    ) -> Result<(), AllocError> {
        let stored = source.fork_prefix(pool, source.len())?;
        if let Some(previous) = self.entries.insert(key, stored) {
            previous.release(pool)?;
        }
        Ok(())
    }

    /// Fork `prefix_len` tokens from the table stored at `key`.
    ///
    /// `Ok(None)` is a miss: the key is absent, or the cached table is
    /// shorter than `prefix_len`. The cache is unchanged on that path.
    /// `Ok(Some)` is a hit. The caller owns the forked table and releases it.
    pub fn lookup(
        &mut self,
        pool: &mut BlockPool,
        key: u64,
        prefix_len: u32,
    ) -> Result<Option<BlockTable>, AllocError> {
        let outcome = match self.entries.get(&key) {
            None => Lookup::Miss,
            Some(cached) if prefix_len > cached.len() => Lookup::Miss,
            Some(cached) => Lookup::Forked(cached.fork_prefix(pool, prefix_len)),
        };
        match outcome {
            Lookup::Miss => {
                self.misses += 1;
                Ok(None)
            }
            Lookup::Forked(Err(err)) => Err(err),
            Lookup::Forked(Ok(table)) => {
                self.hits += 1;
                Ok(Some(table))
            }
        }
    }

    /// Release every cached table.
    ///
    /// Later tables are still released if an earlier one fails, and the
    /// first error is returned. The cache is empty either way.
    pub fn clear(&mut self, pool: &mut BlockPool) -> Result<(), AllocError> {
        let mut first_err = None;
        for (_, table) in self.entries.drain() {
            if let Err(err) = table.release(pool) {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

enum Lookup {
    Miss,
    Forked(Result<BlockTable, AllocError>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_miss_then_a_hit_shares_blocks_and_records_the_rate() {
        let mut pool = BlockPool::new(8, 4);
        let mut parent = BlockTable::new();
        parent.grow(&mut pool, 8).unwrap();
        let parent_blocks: Vec<_> = parent.blocks().to_vec();

        let mut cache = PrefixCache::new();
        assert_eq!(cache.hit_rate(), 0.0);
        assert!(cache.lookup(&mut pool, 7, 8).unwrap().is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);

        cache.insert(&mut pool, 7, &parent).unwrap();
        assert_eq!(cache.len(), 1);

        let child = cache.lookup(&mut pool, 7, 8).unwrap().expect("cached");
        assert_eq!(child.blocks(), parent_blocks.as_slice());
        assert_eq!(pool.refcount(parent_blocks[0]).unwrap(), 3);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hit_rate(), 0.5);

        child.release(&mut pool).unwrap();
        parent.release(&mut pool).unwrap();
        cache.clear(&mut pool).unwrap();
        assert!(cache.is_empty());
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn a_cached_table_that_is_too_short_counts_as_a_miss() {
        let mut pool = BlockPool::new(4, 4);
        let mut parent = BlockTable::new();
        parent.grow(&mut pool, 4).unwrap();

        let mut cache = PrefixCache::new();
        cache.insert(&mut pool, 1, &parent).unwrap();
        assert!(cache.lookup(&mut pool, 1, 8).unwrap().is_none());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1);
        assert_eq!(pool.refcount(parent.blocks()[0]).unwrap(), 2);

        parent.release(&mut pool).unwrap();
        cache.clear(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn the_cache_keeps_blocks_after_the_source_is_released() {
        let mut pool = BlockPool::new(4, 4);
        let mut parent = BlockTable::new();
        parent.grow(&mut pool, 4).unwrap();

        let mut cache = PrefixCache::new();
        cache.insert(&mut pool, 3, &parent).unwrap();
        parent.release(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity() - 1);

        let child = cache
            .lookup(&mut pool, 3, 4)
            .unwrap()
            .expect("still cached");
        child.release(&mut pool).unwrap();
        cache.clear(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }

    #[test]
    fn replacing_a_key_releases_the_previous_blocks() {
        let mut pool = BlockPool::new(4, 4);
        let mut first = BlockTable::new();
        first.grow(&mut pool, 4).unwrap();
        let mut second = BlockTable::new();
        second.grow(&mut pool, 4).unwrap();
        let second_block = second.blocks()[0];

        let mut cache = PrefixCache::new();
        cache.insert(&mut pool, 1, &first).unwrap();
        cache.insert(&mut pool, 1, &second).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(pool.refcount(first.blocks()[0]).unwrap(), 1);
        assert_eq!(pool.refcount(second_block).unwrap(), 2);

        first.release(&mut pool).unwrap();
        second.release(&mut pool).unwrap();
        cache.clear(&mut pool).unwrap();
        assert_eq!(pool.free_count(), pool.capacity());
    }
}
