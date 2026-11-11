# paged-kv

A Rust block pool for a paged KV cache. It tracks which fixed-size blocks are free, which are shared, and how a sequence's tokens map onto them.

The pool does not store tensors. The caller owns the KV bytes and addresses them with a `BlockId`.

This is v0.1: the block manager only. There is no scheduler, no benchmark, and no Python binding.

## What works

`BlockPool` allocates fixed-size blocks, recycles them from an explicit free list, counts reference-counted owners, and copy-on-writes a shared block when one owner must diverge. `BlockTable` grows a sequence and forks a prefix under those rules.

`cargo test` covers that behavior. The claims below name the tests that lock them.

## What does not work

- No continuous batching scheduler, and no sequence queue.
- No measured allocation or scheduling numbers. Nothing here is a performance claim.
- No KV tensor storage, attention kernel, or model runner.
- No Python bindings.
- No preemption hook, prefix-hit metric, or multi-device cache.

Dropping a `BlockTable` without `release` leaks block references until the pool is dropped. The type is marked `must_use`, and the tests call `release`.

## Memory layout

`BlockPool::new(num_blocks, block_size)` builds three parallel structures, indexed by `BlockId`:

| State | Meaning |
| --- | --- |
| `refcounts[id]` | Live owners. `0` means the block is free. |
| `used[id]` | Initialized token slots in that block. `0` when the block is free, and at most `block_size` when it is live. |
| `free_list` | Stack of free ids. `allocate` pops it. `free` pushes an id only when its refcount hits zero. |

A fresh pool is stacked from high ids down to zero, so the first allocations are `0, 1, 2, ...` (`most_recently_freed_block_is_reused_first`). After that, the most recently freed id is handed out next.

One block represents `block_size` token slots. Those bytes are not in this crate. A block id is valid only in the pool that issued it. Copying a `BlockId` does not change its refcount (`shared_block_stays_live_until_the_last_owner_frees_it`).

Blocks are interchangeable. A hole on the free list is reusable for any request, so the pool has no external fragmentation. The waste it reports is internal: `block_size - used` on each live block. `wasted_slots` sums that once per physical block, including when several owners share it (`shared_owners_do_not_double_count_tail_waste`). An idle pool reports zero (`idle_pool_has_zero_fragmentation`). A freshly allocated block starts at `used == 0`, so it counts as fully wasted until the caller fills it (`partial_block_reports_tail_waste`).

## Ownership

- `allocate` returns one block with refcount 1.
- `acquire` adds an owner of the current contents. It fails on a free or unknown id (`acquire_rejects_a_free_or_unknown_block`).
- `free` drops one owner. The initialized length stays until the last owner frees the block. The last free pushes the id and clears `used` (`freeing_a_partial_block_drops_its_waste`). A second free of an already free block is `NotAllocated` and does not push a duplicate (`double_free_is_rejected_and_does_not_grow_the_free_list`).
- `cow` returns the same id when the refcount is 1 (`cow_keeps_a_unique_block_in_place`). When the block is shared, it allocates a new block, copies the initialized length, and drops one reference to the original. The other owners still see the original length (`cow_forks_a_shared_block_without_changing_the_other_owner`). The caller copies KV bytes only when the returned id differs. If the pool cannot allocate the fork, the shared block is unchanged (`cow_of_a_shared_block_fails_cleanly_when_the_pool_is_full`).
- `BlockTable` holds one reference per block it lists. It does not own the pool. `release` drops those references. Other tables that still hold a block keep it allocated (`fork_on_a_block_boundary_shares_only_the_covered_blocks`).

## How a prefix fork shares blocks

`BlockTable::grow` appends tokens. Full blocks are marked fully used. The tail holds the remainder (`grow_fills_full_blocks_and_a_partial_tail`). `grow` does not shorten a table (`grow_rejects_a_shorter_length_and_an_allocation_it_cannot_finish`). If the pool cannot cover the whole request, the table and the pool stay as they were.

`fork_prefix(prefix_len)` builds another table for the first `prefix_len` tokens:

- A prefix that ends on a block boundary shares those full blocks and leaves the parent's later blocks private.
- A prefix that covers the whole table also shares a partial tail. The next append into that tail copy-on-writes it (`fork_of_the_whole_sequence_shares_the_tail_until_append`). If that copy cannot be allocated, neither table changes (`extending_a_shared_tail_fails_cleanly_when_cow_cannot_allocate`).
- A prefix that stops inside a block the parent has already written past shares only the full blocks before that point. The child gets a new private block with `used` set to the partial length, so it does not alias the parent's later tokens (`fork_inside_a_written_block_gets_a_private_boundary`). If that private block cannot be allocated, the acquires are rolled back (`fork_rolls_back_acquires_when_the_private_block_cannot_be_allocated`).
- A zero-length prefix allocates nothing. A prefix past the table length returns `PrefixOutOfRange` and changes no refcount (`empty_and_overlong_prefixes_do_not_change_refcounts`).

Full blocks are not written in place. Extending past a full block allocates a new block, including after a boundary fork.

```rust
use paged_kv::{BlockPool, BlockTable};

fn share_a_prefix() -> Result<(), paged_kv::AllocError> {
    let mut pool = BlockPool::new(8, 16);
    let mut parent = BlockTable::new();
    parent.grow(&mut pool, 20)?;

    // Same contents, including the partial tail: both tables hold those ids.
    let mut child = parent.fork_prefix(&mut pool, 20)?;
    // The next token copy-on-writes the shared tail. The parent keeps the old id.
    child.grow(&mut pool, 21)?;

    child.release(&mut pool)?;
    parent.release(&mut pool)?;
    Ok(())
}
```

## Tests

```sh
cargo test
```

v0.1 stops at this block manager. Scheduler behavior and any comparison against a non-paged baseline are later work, and they are not claimed here.
