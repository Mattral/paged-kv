# v0.1.0

Notes for the existing tag `v0.1.0` (`01bbcbf`). Do not publish a new crate. This file does not create the GitHub release.

Paged KV-cache blocks and an FCFS scheduler.

That tag reports a synthetic packing mix: 607 paged sequences, 529 when each buffer doubles and 64 when every sequence reserves 1,024 slots. The budget is 65,536 token slots.

```sh
cargo add paged-kv
```

https://crates.io/crates/paged-kv

One sequence or a small batch on CPU `f32`. There is no model server and no second model.

`main` later measured the Azure conversation trace on the same budget, which is 4.5 GiB at 73,728 bytes per GPT-2 small token. Paged blocks fit 75 sequences, doubling fits 50 and a max-length reserve of 14,089 fits 4. Those commits are not in tag `v0.1.0`.
