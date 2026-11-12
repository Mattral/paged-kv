//! Non-paged baseline: one contiguous token span per sequence.
//!
//! There is no block size, free list, refcount, or prefix sharing. Capacity
//! is a single integer. [`ContiguousTable::reserve_exact`] sizes it once.
//! [`ContiguousTable::grow`] doubles it, the way a normal buffer grows.
//! Nothing here stores KV bytes.

use crate::AllocError;

/// Logical length and one capacity, with no page metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContiguousTable {
    len: u32,
    capacity: u32,
}

impl Default for ContiguousTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ContiguousTable {
    /// Empty span. Capacity is zero until `grow` or `reserve_exact`.
    pub fn new() -> Self {
        Self {
            len: 0,
            capacity: 0,
        }
    }

    /// Tokens currently represented.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// `true` when no tokens are represented.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reserved token slots. This may be larger than [`ContiguousTable::len`]
    /// after a doubling grow.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Make room for exactly `tokens` slots.
    ///
    /// Does not change the logical length. A later `grow` up to `tokens`
    /// does not increase capacity again. Shrinking capacity is not supported.
    pub fn reserve_exact(&mut self, tokens: u32) {
        if tokens > self.capacity {
            self.capacity = tokens;
        }
    }

    /// Set the logical length to `new_len`.
    ///
    /// If that passes the capacity, capacity doubles until it fits, starting
    /// from at least one slot. The length is unchanged when this returns an
    /// error.
    pub fn grow(&mut self, new_len: u32) -> Result<(), AllocError> {
        if new_len < self.len {
            return Err(AllocError::LengthRegression {
                current: self.len,
                requested: new_len,
            });
        }
        if new_len > self.capacity {
            let mut cap = self.capacity.max(1);
            while cap < new_len {
                let next = cap.saturating_mul(2);
                if next == cap {
                    cap = new_len;
                    break;
                }
                cap = next;
            }
            self.capacity = cap;
        }
        self.len = new_len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grow_doubles_capacity_until_the_length_fits() {
        let mut table = ContiguousTable::new();
        table.grow(1).unwrap();
        assert_eq!(table.len(), 1);
        assert_eq!(table.capacity(), 1);

        table.grow(3).unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.capacity(), 4);

        table.grow(4).unwrap();
        assert_eq!(table.capacity(), 4);
    }

    #[test]
    fn reserve_exact_does_not_double_past_the_request() {
        let mut table = ContiguousTable::new();
        table.reserve_exact(100);
        assert_eq!(table.len(), 0);
        assert_eq!(table.capacity(), 100);

        table.grow(100).unwrap();
        assert_eq!(table.len(), 100);
        assert_eq!(table.capacity(), 100);

        table.reserve_exact(40);
        assert_eq!(table.capacity(), 100);
    }

    #[test]
    fn shrink_is_rejected_and_leaves_the_span_unchanged() {
        let mut table = ContiguousTable::new();
        table.grow(5).unwrap();
        let before = table.clone();
        assert_eq!(
            table.grow(2).unwrap_err(),
            AllocError::LengthRegression {
                current: 5,
                requested: 2,
            }
        );
        assert_eq!(table, before);
    }
}
