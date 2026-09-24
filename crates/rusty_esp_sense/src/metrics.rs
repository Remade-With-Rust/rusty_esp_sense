//! What a result is judged by.

/// The fraction of `pred` equal to `truth`.
#[must_use]
pub fn accuracy(pred: &[usize], truth: &[usize]) -> f64 {
    if pred.is_empty() {
        return 0.0;
    }
    pred.iter().zip(truth).filter(|(a, b)| a == b).count() as f64 / pred.len() as f64
}

/// `m[truth][pred]` counts over `k` classes.
#[must_use]
pub fn confusion(pred: &[usize], truth: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut m = vec![vec![0usize; k]; k];
    for (&p, &t) in pred.iter().zip(truth) {
        if p < k && t < k {
            m[t][p] += 1;
        }
    }
    m
}

/// Torso-normalised PCK: the fraction of joints within `threshold` times the
/// torso length (the distance between joints `torso.0` and `torso.1` of the
/// ground truth) of where they are, averaged over poses. MultiFormer's
/// definition, which RuView's MM-Fi numbers use; `threshold` 0.2 is PCK@20.
/// A pose whose torso has no length is skipped rather than divided by zero.
#[must_use]
pub fn pck(
    pred: &[Vec<[f32; 2]>],
    truth: &[Vec<[f32; 2]>],
    torso: (usize, usize),
    threshold: f32,
) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for (p, t) in pred.iter().zip(truth) {
        let (Some(a), Some(b)) = (t.get(torso.0), t.get(torso.1)) else {
            continue;
        };
        let torso_len = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        if torso_len <= f32::EPSILON {
            continue;
        }
        for (q, r) in p.iter().zip(t) {
            let d = ((q[0] - r[0]).powi(2) + (q[1] - r[1]).powi(2)).sqrt();
            total += 1;
            if d <= threshold * torso_len {
                hit += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hit as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_and_confusion_count_what_they_say() {
        let p = [0, 1, 1, 0];
        let t = [0, 1, 0, 0];
        assert!((accuracy(&p, &t) - 0.75).abs() < 1e-12);
        assert_eq!(confusion(&p, &t, 2), vec![vec![2, 1], vec![0, 1]]);
        assert_eq!(accuracy(&[], &[]), 0.0);
    }

    #[test]
    fn pck_counts_joints_inside_a_fraction_of_the_torso() {
        // torso from joint 0 (0,0) to joint 1 (0,10): length 10, PCK@20 radius 2.
        let truth = vec![vec![[0.0, 0.0], [0.0, 10.0], [5.0, 5.0], [8.0, 8.0]]];
        let pred = vec![vec![[0.0, 1.9], [0.0, 10.0], [5.0, 7.5], [8.0, 8.0]]];
        assert!((pck(&pred, &truth, (0, 1), 0.2) - 0.75).abs() < 1e-12);
        let flat = vec![vec![[1.0, 1.0], [1.0, 1.0]]];
        assert_eq!(pck(&flat, &flat, (0, 1), 0.2), 0.0, "no torso, no score");
    }
}
