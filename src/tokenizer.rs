//! Byte vocabulary.
//!
//! Token `b` is the raw byte `b`, for `b < 256`. [`EOS`] is an extra id that
//! [`encode`] does not emit. This is not SentencePiece and not a trained BPE
//! model. Any UTF-8 string round-trips through [`encode`] and [`decode`].

/// Number of ids: 256 bytes plus [`EOS`].
pub const VOCAB_SIZE: u32 = 257;

/// End-of-sequence id. It is not a byte.
pub const EOS: u32 = 256;

/// Token ids for the bytes in `text`.
pub fn encode(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// Bytes for `ids`, skipping [`EOS`] and any id outside the byte range.
///
/// Invalid UTF-8 is replaced with U+FFFD. A string that came from [`encode`]
/// comes back unchanged.
pub fn decode(ids: &[u32]) -> String {
    let bytes = ids
        .iter()
        .copied()
        .filter_map(|id| u8::try_from(id).ok())
        .collect::<Vec<u8>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_then_decode_round_trips() {
        assert_eq!(encode(""), Vec::<u32>::new());
        assert_eq!(decode(&[]), "");
        assert_eq!(encode("hi"), vec![104, 105]);
        let text = "héllo\n";
        assert_eq!(decode(&encode(text)), text);
        assert_eq!(decode(&[104, EOS, 105]), "hi");
        assert_eq!(decode(&[104, VOCAB_SIZE, 105]), "hi");
    }
}
