//! Qwen sparse attention (QSA) indexer: per-block scores and the visible-token selection for
//! one query. Mirrors llama.cpp `build_qsa_top_k` / `set_input_qsa` (single sequence):
//!   score[b] = sum_h relu(q_h . pk_b)           (pk_b = rope(rmsnorm(mean of the block's raw keys)))
//!   width    = min(n_kv, top_k + r - 1)         (tokens visible to the query)
//!   the incomplete tail block [n_kv/r*r, n_kv) is always visible; the remaining slots go to whole
//!   blocks by descending score (ties: lowest block index) and, when slots % r != 0, to the first
//!   slots % r tokens of the next-best block (llama.cpp's top_k picks arbitrary members there).
use crate::attn::dot;

/// scores[b] = sum_h relu(q_h . pk[b]) for b in b0..b1. q: [n_heads][d], pk: [>= b1][d].
pub fn score_blocks(q: &[f32], pk: &[f32], d: usize, b0: usize, b1: usize, scores: &mut [f32]) {
    let nh = q.len() / d;
    for b in b0..b1 {
        let kb = &pk[b * d..(b + 1) * d];
        let mut s = 0f32;
        for h in 0..nh {
            let v = dot(&q[h * d..(h + 1) * d], kb);
            if v > 0.0 {
                s += v;
            }
        }
        scores[b] = s;
    }
}

/// Number of complete (poolable) blocks below the tail for a query at position `pos`.
#[inline]
pub fn n_complete_blocks(pos: usize, r: usize) -> usize {
    (pos + 1) / r
}

/// True when the query at `pos` sees fewer than all `pos + 1` cached tokens.
#[inline]
pub fn is_sparse(pos: usize, r: usize, top_k: usize) -> bool {
    pos + 1 > top_k + r - 1
}

/// Visible token ranges (start, len), sorted and merged, for the query at `pos`.
/// `scores` holds one score per complete block (len >= n_complete_blocks(pos, r)).
/// Returns false (single full range) when the query is dense.
pub fn select_ranges(scores: &[f32], pos: usize, r: usize, top_k: usize, idx: &mut Vec<u32>, ranges: &mut Vec<(u32, u32)>) -> bool {
    let n_kv = pos + 1;
    ranges.clear();
    if !is_sparse(pos, r, top_k) {
        ranges.push((0, n_kv as u32));
        return false;
    }
    let width = top_k + r - 1;
    let tail_start = n_kv / r * r;
    let tail = n_kv - tail_start;
    let nb = tail_start / r;
    let slots = width - tail;
    let whole = slots / r;
    let part = slots % r;
    let need = whole + (part > 0) as usize;
    debug_assert!(need <= nb);
    let rank = |a: u32, b: u32| scores[b as usize].partial_cmp(&scores[a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b));
    idx.clear();
    idx.extend(0..nb as u32);
    if need == 0 {
        idx.clear();
    } else if need < nb {
        idx.select_nth_unstable_by(need - 1, |&a, &b| rank(a, b));
        idx.truncate(need);
    }
    let part_blk = if part > 0 { idx.iter().copied().max_by(|&a, &b| rank(a, b)).unwrap_or(u32::MAX) } else { u32::MAX };
    idx.sort_unstable();
    let push = |ranges: &mut Vec<(u32, u32)>, start: u32, len: u32| {
        if len == 0 {
            return;
        }
        if let Some(last) = ranges.last_mut() {
            if last.0 + last.1 == start {
                last.1 += len;
                return;
            }
        }
        ranges.push((start, len));
    };
    for &b in idx.iter() {
        let len = if b == part_blk { part } else { r };
        push(ranges, b * r as u32, len as u32);
    }
    push(ranges, tail_start as u32, tail as u32);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_below_width() {
        let (mut idx, mut rg) = (Vec::new(), Vec::new());
        assert!(!select_ranges(&[], 2050, 4, 2048, &mut idx, &mut rg));
        assert_eq!(rg, vec![(0, 2051)]);
    }
    #[test]
    fn sparse_picks_best_blocks_and_tail() {
        // r = 4, top_k = 8: width 11. pos = 22 -> n_kv 23, tail_start 20 (tail 3), 5 complete blocks,
        // slots 8 -> 2 whole blocks, no partial.
        let scores = [0.5, 3.0, 0.0, 3.0, 1.0];
        let (mut idx, mut rg) = (Vec::new(), Vec::new());
        assert!(select_ranges(&scores, 22, 4, 8, &mut idx, &mut rg));
        assert_eq!(rg, vec![(4, 4), (12, 4), (20, 3)]);
        // pos = 23 -> n_kv 24, tail 0, 6 blocks, slots 11 -> 2 whole + 3 of the third best (block 4)
        let scores = [0.5, 3.0, 0.0, 3.0, 1.0, 0.7];
        assert!(select_ranges(&scores, 23, 4, 8, &mut idx, &mut rg));
        assert_eq!(rg, vec![(4, 4), (12, 7)]); // 12..16 whole, 16..19 partial (merged)
        // ties: lowest index wins
        let scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert!(select_ranges(&scores, 23, 4, 8, &mut idx, &mut rg));
        assert_eq!(rg, vec![(0, 11)]);
    }
    #[test]
    fn scores_relu_sum() {
        let q = [1.0, 0.0, -1.0, 0.0]; // 2 heads, d = 2
        let pk = [1.0, 1.0, -2.0, 5.0];
        let mut s = [0f32; 2];
        score_blocks(&q, &pk, 2, 0, 2, &mut s);
        assert_eq!(s, [1.0, 2.0]);
    }
}
