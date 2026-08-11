use crate::engine::EngineError;

pub(crate) fn find_seam(
    energy: &[f32],
    height: usize,
    width: usize,
) -> Result<Vec<usize>, EngineError> {
    let costs = cumulative_costs(energy, height, width);
    let last_row = (height - 1) * width;
    let column = argmin(&costs[last_row..last_row + width]);

    if !costs[last_row + column].is_finite() {
        return Err(EngineError::NoSeam);
    }

    let mut seam = vec![0; height];
    seam[height - 1] = column;

    for row in (0..height - 1).rev() {
        let next_column = seam[row + 1];
        let start = next_column.saturating_sub(1);
        let end = (next_column + 2).min(width);
        let row_start = row * width;
        let prev_col =
            argmin(&costs[row_start + start..row_start + end]) + start;

        if !costs[row_start + prev_col].is_finite() {
            return Err(EngineError::NoSeam);
        }
        seam[row] = prev_col;
    }

    Ok(seam)
}

fn cumulative_costs(energy: &[f32], height: usize, width: usize) -> Vec<f64> {
    let mut costs: Vec<f64> =
        energy.iter().map(|value| f64::from(*value)).collect();

    for row in 1..height {
        let prev_start = (row - 1) * width;
        let curr_start = row * width;

        for column in 0..width {
            let start = column.saturating_sub(1);
            let end = (column + 2).min(width);
            let prev_col =
                argmin(&costs[prev_start + start..prev_start + end]) + start;
            costs[curr_start + column] += costs[prev_start + prev_col];
        }
    }

    costs
}

pub(crate) fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] < values[best] {
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::gradient_energy;

    #[test]
    fn one_seam_is_connected() {
        let image = vec![0; 4 * 5 * crate::CHANNELS];
        let energy = gradient_energy(&image, 4, 5);
        let seam = find_seam(&energy, 4, 5).unwrap();

        assert_eq!(seam.len(), 4);
        assert!(seam.windows(2).all(|pair| pair[0].abs_diff(pair[1]) <= 1));
    }
}
