//! Paged KV-cache block management.
//!
//! The pool tracks fixed-size blocks for a key/value cache.
//! Callers own the KV bytes and address them by block id.
