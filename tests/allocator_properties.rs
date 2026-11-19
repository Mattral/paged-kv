//! Seeded allocator checks. No fuzzer crate: `next_u64` is a fixed xorshift.

use paged_kv::{AllocError, BlockId, BlockPool, BlockTable};

const SEED: u64 = 0xA11C_5EED;
const STEPS: u32 = 2_048;
const BLOCKS: u32 = 16;
const BLOCK_SIZE: u32 = 16;

#[test]
fn allocator_refcounts_stay_non_negative_under_a_seeded_mix() {
    let mut rng = SEED;
    let mut pool = BlockPool::new(BLOCKS, BLOCK_SIZE);
    let mut extras: Vec<BlockId> = Vec::new();
    let mut tables: Vec<BlockTable> = Vec::new();

    for step in 0..STEPS {
        match next_u64(&mut rng) % 4 {
            0 => allocate(&mut rng, &mut pool, &mut extras, &mut tables),
            1 => free(&mut rng, &mut pool, &mut extras, &tables),
            2 => share(&mut rng, &mut pool, &mut extras, &mut tables),
            _ => release(&mut rng, &mut pool, &mut tables),
        }
        check(step, &pool, &extras, &tables);
    }

    while let Some(table) = tables.pop() {
        table
            .release(&mut pool)
            .expect("table still owns its blocks");
        check(STEPS, &pool, &extras, &tables);
    }
    while let Some(id) = extras.pop() {
        pool.free(id).expect("extra owner");
        check(STEPS, &pool, &extras, &tables);
    }
    assert_eq!(pool.free_count(), pool.capacity());
}

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn allocate(
    rng: &mut u64,
    pool: &mut BlockPool,
    extras: &mut Vec<BlockId>,
    tables: &mut Vec<BlockTable>,
) {
    if next_u64(rng) % 2 == 0 {
        match pool.allocate() {
            Ok(id) => extras.push(id),
            Err(AllocError::Exhausted) => assert_eq!(pool.free_count(), 0),
            Err(err) => panic!("allocate failed: {err}"),
        }
        return;
    }

    let mut table = BlockTable::new();
    let len = 1 + (next_u64(rng) as u32) % (BLOCK_SIZE * 4);
    match table.grow(pool, len) {
        Ok(()) => tables.push(table),
        Err(AllocError::Exhausted) => assert!(table.is_empty(), "a failed grow rolls back"),
        Err(err) => panic!("grow failed: {err}"),
    }
}

fn free(rng: &mut u64, pool: &mut BlockPool, extras: &mut Vec<BlockId>, tables: &[BlockTable]) {
    if !extras.is_empty() {
        let index = (next_u64(rng) as usize) % extras.len();
        let id = extras.swap_remove(index);
        pool.free(id).expect("recorded owner");
        return;
    }

    let index = (next_u64(rng) as u32) % pool.capacity();
    let tally = owner_tally(extras, tables, pool.capacity());
    if tally[index as usize] == 0 {
        let err = pool
            .free(BlockId::from_index(index))
            .expect_err("a free block has no owner");
        assert!(matches!(err, AllocError::NotAllocated(_)));
    }
}

fn share(
    rng: &mut u64,
    pool: &mut BlockPool,
    extras: &mut Vec<BlockId>,
    tables: &mut Vec<BlockTable>,
) {
    let prefer_fork = next_u64(rng) % 2 == 0;
    if prefer_fork {
        if !fork_one(rng, pool, tables) {
            acquire_one(rng, pool, extras, tables);
        }
    } else if !acquire_one(rng, pool, extras, tables) {
        fork_one(rng, pool, tables);
    }
}

fn fork_one(rng: &mut u64, pool: &mut BlockPool, tables: &mut Vec<BlockTable>) -> bool {
    let candidates: Vec<usize> = tables
        .iter()
        .enumerate()
        .filter(|(_, table)| !table.is_empty())
        .map(|(index, _)| index)
        .collect();
    if candidates.is_empty() {
        return false;
    }
    let index = candidates[(next_u64(rng) as usize) % candidates.len()];
    let prefix = 1 + (next_u64(rng) as u32) % tables[index].len();
    match tables[index].fork_prefix(pool, prefix) {
        Ok(child) => {
            tables.push(child);
            true
        }
        Err(AllocError::Exhausted) => true,
        Err(err) => panic!("fork failed: {err}"),
    }
}

fn acquire_one(
    rng: &mut u64,
    pool: &mut BlockPool,
    extras: &mut Vec<BlockId>,
    tables: &[BlockTable],
) -> bool {
    let tally = owner_tally(extras, tables, pool.capacity());
    let live: Vec<u32> = tally
        .iter()
        .enumerate()
        .filter(|(_, rc)| **rc > 0)
        .map(|(index, _)| index as u32)
        .collect();
    if live.is_empty() {
        return false;
    }
    let index = live[(next_u64(rng) as usize) % live.len()];
    let id = BlockId::from_index(index);
    match pool.acquire(id) {
        Ok(_) => {
            extras.push(id);
            true
        }
        Err(err) => panic!("live block {id} refused acquire: {err}"),
    }
}

fn release(rng: &mut u64, pool: &mut BlockPool, tables: &mut Vec<BlockTable>) {
    if tables.is_empty() {
        return;
    }
    let index = (next_u64(rng) as usize) % tables.len();
    let table = tables.swap_remove(index);
    table.release(pool).expect("table owns its blocks");
}

fn owner_tally(extras: &[BlockId], tables: &[BlockTable], blocks: u32) -> Vec<i32> {
    let mut tally = vec![0i32; blocks as usize];
    for id in extras {
        add_owner(&mut tally, *id);
    }
    for table in tables {
        for id in table.blocks() {
            add_owner(&mut tally, *id);
        }
    }
    tally
}

fn add_owner(tally: &mut [i32], id: BlockId) {
    let index = id.index() as usize;
    tally[index] = tally[index].checked_add(1).expect("owner tally overflowed");
}

fn check(step: u32, pool: &BlockPool, extras: &[BlockId], tables: &[BlockTable]) {
    let tally = owner_tally(extras, tables, pool.capacity());
    let mut live = 0u32;
    for (index, rc) in tally.iter().enumerate() {
        assert!(
            *rc >= 0,
            "step {step}: block {index} refcount {rc} is negative"
        );
        let pool_rc = pool
            .refcount(BlockId::from_index(index as u32))
            .expect("id is inside the pool");
        let signed = i32::try_from(pool_rc).unwrap_or_else(|_| {
            panic!("step {step}: block {index} refcount {pool_rc} is not a signed count")
        });
        assert!(
            signed >= 0,
            "step {step}: block {index} refcount is negative"
        );
        assert_eq!(signed, *rc, "step {step}: block {index}");
        if *rc > 0 {
            live += 1;
        }
    }
    assert_eq!(
        pool.free_count() + live,
        pool.capacity(),
        "step {step}: free {} + live {live}",
        pool.free_count()
    );
}
