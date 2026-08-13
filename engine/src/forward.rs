//! Forward-energy seam search.
//!
//! Transition costs are computed from neighboring pixels in the current
//! working image. Dynamic programming then chooses one connected vertical seam
//! before compaction changes the image for the next iteration.

use crate::engine::EngineError;
use crate::image::pixel;
use crate::seam::argmin;

pub(crate) fn find_seam(
    image: &[u8],
    height: usize,
    width: usize,
) -> Result<Vec<usize>, EngineError> {
    let (costs, preds) = cumulative_costs(image, height, width);
    let last_row = (height - 1) * width;
    let column = argmin(&costs[last_row..last_row + width]);

    if !costs[last_row + column].is_finite() {
        return Err(EngineError::NoSeam);
    }

    let mut seam = vec![0; height];
    seam[height - 1] = column;

    for row in (1..height).rev() {
        seam[row - 1] = preds[row * width + seam[row]];
    }

    Ok(seam)
}

fn cumulative_costs(
    image: &[u8],
    height: usize,
    width: usize,
) -> (Vec<f64>, Vec<usize>) {
    let mut costs = vec![0.0; height * width];
    let mut preds = vec![0; height * width];

    for row in 1..height {
        let prev_start = (row - 1) * width;
        let curr_start = row * width;
        let (left_costs, upward_costs, right_costs) =
            transition_costs(image, width, row);

        for column in 0..width {
            let mut best = costs[prev_start + column] + upward_costs[column];
            let mut prev_col = column;

            if column > 0 {
                let candidate =
                    costs[prev_start + column - 1] + left_costs[column];
                if candidate <= best {
                    best = candidate;
                    prev_col = column - 1;
                }
            }
            if column + 1 < width {
                let candidate =
                    costs[prev_start + column + 1] + right_costs[column];
                if candidate < best {
                    best = candidate;
                    prev_col = column + 1;
                }
            }

            costs[curr_start + column] = best;
            preds[curr_start + column] = prev_col;
        }
    }

    (costs, preds)
}

fn transition_costs(
    image: &[u8],
    width: usize,
    row: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut left_costs = vec![0.0; width];
    let mut upward_costs = vec![0.0; width];
    let mut right_costs = vec![0.0; width];

    for column in 0..width {
        let current_left = pixel(image, width, row, column.saturating_sub(1));
        let current_right =
            pixel(image, width, row, (column + 1).min(width - 1));
        let previous = pixel(image, width, row - 1, column);
        let upward = color_distance(current_right, current_left);

        left_costs[column] = upward + color_distance(previous, current_left);
        upward_costs[column] = upward;
        right_costs[column] = upward + color_distance(previous, current_right);
    }

    (left_costs, upward_costs, right_costs)
}

fn color_distance(left: &[u8], right: &[u8]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| f64::from(left.abs_diff(right)))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_seam_matches_reference_path() {
        let image = vec![
            58, 90, 92, 84, 109, 255, 104, 251, 115, 232, 45, 138, 40, 40, 255,
            216, 21, 220, 210, 61, 203, 216, 102, 108, 5, 17, 73, 186, 139,
            170, 81, 0, 7, 101, 227, 165,
        ];

        let seam = find_seam(&image, 4, 3).unwrap();
        let (costs, _) = cumulative_costs(&image, 4, 3);

        assert_eq!(seam, vec![1, 1, 0, 1]);
        assert_eq!(costs[10], 674.0);
    }

    #[test]
    fn find_seam_handles_single_column() {
        let image = vec![0; 4 * crate::CHANNELS];
        let seam = find_seam(&image, 4, 1).unwrap();

        assert_eq!(seam, vec![0, 0, 0, 0]);
    }

    #[test]
    fn find_seam_prefers_leftmost_ties() {
        let image = vec![0; 3 * 4 * crate::CHANNELS];
        let seam = find_seam(&image, 3, 4).unwrap();

        assert_eq!(seam, vec![0, 0, 0]);
    }
}
