#[derive(Debug)]
pub struct DelayedPatternMaskIds<const N: usize> {
    batches: [Vec<i64>; N],
}

impl<const N: usize> DelayedPatternMaskIds<N> {
    pub fn new() -> Self {
        assert!(N > 0, "N must be greater than 0");
        Self {
            batches: std::array::from_fn(|_| Vec::new()),
        }
    }

    pub fn push(&mut self, token_ids: impl IntoIterator<Item = i64>) {
        let mut iter = token_ids.into_iter();
        for i in 0..N {
            let Some(id) = iter.next() else {
                panic!("Expected exactly {N} token_ids, got fewer");
            };
            self.batches[i].push(id);
        }
        if iter.next().is_some() {
            panic!("Expected exactly {N} token_ids, got more");
        }
    }

    /// Returns the last tokens, progressively delayed by position.
    pub fn last_delayed_masked(&self, pad_token_id: i64) -> [i64; N] {
        let seq_len = self.batches[0].len();
        std::array::from_fn(|i| {
            if (seq_len as isize - i as isize) <= 0 {
                pad_token_id
            } else {
                *self.batches[i].last().expect("No input_ids found")
            }
        })
    }

    /// Returns last diagonal values across batches if enough data exists.
    pub fn last_de_delayed(&self) -> Option<[i64; N]> {
        if self.batches[0].len() < N {
            return None;
        }
        Some(std::array::from_fn(|i| self.batches[i][self.batches[i].len() - N + i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last_delayed_masked() {
        let mut ids = DelayedPatternMaskIds::<4>::new();
        assert_eq!(ids.last_delayed_masked(0), [0, 0, 0, 0]);
        ids.push([1, 2, 3, 4]);
        assert_eq!(ids.last_delayed_masked(0), [1, 0, 0, 0]);
        ids.push([5, 6, 7, 8]);
        assert_eq!(ids.last_delayed_masked(0), [5, 6, 0, 0]);
        ids.push([9, 10, 11, 12]);
        assert_eq!(ids.last_delayed_masked(0), [9, 10, 11, 0]);
        ids.push([13, 14, 15, 16]);
        assert_eq!(ids.last_delayed_masked(0), [13, 14, 15, 16]);
        ids.push([17, 18, 19, 20]);
        assert_eq!(ids.last_delayed_masked(0), [17, 18, 19, 20]);
    }

    #[test]
    fn test_last_de_delayed() {
        let mut ids = DelayedPatternMaskIds::<4>::new();
        assert_eq!(ids.last_de_delayed(), None);
        ids.push([1, 2, 3, 4]);
        assert_eq!(ids.last_de_delayed(), None);
        ids.push([5, 6, 7, 8]);
        assert_eq!(ids.last_de_delayed(), None);
        ids.push([9, 10, 11, 12]);
        assert_eq!(ids.last_de_delayed(), None);
        ids.push([13, 14, 15, 16]);
        assert_eq!(ids.last_de_delayed(), Some([1, 6, 11, 16]));
        ids.push([17, 18, 19, 20]);
        assert_eq!(ids.last_de_delayed(), Some([5, 10, 15, 20]));
    }
}
