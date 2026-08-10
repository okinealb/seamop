use std::error::Error;
use std::fmt;

const CHANNELS: usize = 3;
const BORDER_ENERGY: f32 = 1000.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    EmptyImage,
    InvalidImageLength { expected: usize, actual: usize },
    InvalidTargetWidth { target: usize, source: usize },
    NoSeam,
    SizeOverflow,
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => formatter.write_str("image dimensions must be positive"),
            Self::InvalidImageLength { expected, actual } => write!(
                formatter,
                "image buffer must contain {expected} bytes; got {actual}"
            ),
            Self::InvalidTargetWidth { target, source } => write!(
                formatter,
                "target width must be between 1 and {source}; got {target}"
            ),
            Self::NoSeam => formatter.write_str("no finite seam remains"),
            Self::SizeOverflow => formatter.write_str("image dimensions overflow the buffer size"),
        }
    }
}

impl Error for EngineError {}

#[derive(Debug, PartialEq, Eq)]
pub struct VerticalPlan {
    pub result: Vec<u8>,
    pub removed_mask: Vec<bool>,
    pub source_height: usize,
    pub source_width: usize,
    pub target_width: usize,
}

pub fn plan_vertical_gradient(
    image: &[u8],
    height: usize,
    width: usize,
    target_width: usize,
) -> Result<VerticalPlan, EngineError> {
    let pixel_count = checked_pixel_count(height, width)?;
    let expected_bytes = pixel_count
        .checked_mul(CHANNELS)
        .ok_or(EngineError::SizeOverflow)?;

    if height == 0 || width == 0 {
        return Err(EngineError::EmptyImage);
    }
    if image.len() != expected_bytes {
        return Err(EngineError::InvalidImageLength {
            expected: expected_bytes,
            actual: image.len(),
        });
    }
    if target_width == 0 || target_width > width {
        return Err(EngineError::InvalidTargetWidth {
            target: target_width,
            source: width,
        });
    }

    let mut working = image.to_vec();
    let mut source_indices: Vec<usize> = (0..pixel_count).collect();
    let mut removed_mask = vec![false; pixel_count];
    let mut current_width = width;

    while current_width > target_width {
        let energy = gradient_energy(&working, height, current_width);
        let seam = find_seam(&energy, height, current_width)?;

        for row in 0..height {
            let current_index = row * current_width + seam[row];
            removed_mask[source_indices[current_index]] = true;
        }

        let (next_image, next_indices) =
            compact(&working, &source_indices, height, current_width, &seam);
        working = next_image;
        source_indices = next_indices;
        current_width -= 1;
    }

    Ok(VerticalPlan {
        result: working,
        removed_mask,
        source_height: height,
        source_width: width,
        target_width,
    })
}

fn checked_pixel_count(height: usize, width: usize) -> Result<usize, EngineError> {
    height.checked_mul(width).ok_or(EngineError::SizeOverflow)
}

fn gradient_energy(image: &[u8], height: usize, width: usize) -> Vec<f32> {
    let mut energy = vec![BORDER_ENERGY; height * width];

    for row in 1..height.saturating_sub(1) {
        for column in 1..width.saturating_sub(1) {
            let right = pixel(image, width, row, column + 1);
            let left = pixel(image, width, row, column - 1);
            let below = pixel(image, width, row + 1, column);
            let above = pixel(image, width, row - 1, column);

            let dx_red = f32::from(right[0]) - f32::from(left[0]);
            let dx_green = f32::from(right[1]) - f32::from(left[1]);
            let dx_blue = f32::from(right[2]) - f32::from(left[2]);
            let dy_red = f32::from(below[0]) - f32::from(above[0]);
            let dy_green = f32::from(below[1]) - f32::from(above[1]);
            let dy_blue = f32::from(below[2]) - f32::from(above[2]);

            energy[row * width + column] = (dx_red * dx_red
                + dx_green * dx_green
                + dx_blue * dx_blue
                + dy_red * dy_red
                + dy_green * dy_green
                + dy_blue * dy_blue)
                .sqrt();
        }
    }

    energy
}

fn pixel(image: &[u8], width: usize, row: usize, column: usize) -> &[u8] {
    let start = (row * width + column) * CHANNELS;
    &image[start..start + CHANNELS]
}

fn find_seam(energy: &[f32], height: usize, width: usize) -> Result<Vec<usize>, EngineError> {
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
        let stop = (next_column + 2).min(width);
        let row_start = row * width;
        let previous_column = argmin(&costs[row_start + start..row_start + stop]) + start;

        if !costs[row_start + previous_column].is_finite() {
            return Err(EngineError::NoSeam);
        }
        seam[row] = previous_column;
    }

    Ok(seam)
}

fn cumulative_costs(energy: &[f32], height: usize, width: usize) -> Vec<f64> {
    let mut costs: Vec<f64> = energy.iter().map(|value| f64::from(*value)).collect();

    for row in 1..height {
        let previous_start = (row - 1) * width;
        let current_start = row * width;

        for column in 0..width {
            let start = column.saturating_sub(1);
            let stop = (column + 2).min(width);
            let previous = argmin(&costs[previous_start + start..previous_start + stop]) + start;
            costs[current_start + column] += costs[previous_start + previous];
        }
    }

    costs
}

fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] < values[best] {
            best = index;
        }
    }
    best
}

fn compact(
    image: &[u8],
    source_indices: &[usize],
    height: usize,
    width: usize,
    seam: &[usize],
) -> (Vec<u8>, Vec<usize>) {
    let next_width = width - 1;
    let mut next_image = Vec::with_capacity(height * next_width * CHANNELS);
    let mut next_indices = Vec::with_capacity(height * next_width);

    for row in 0..height {
        for column in 0..width {
            if column == seam[row] {
                continue;
            }

            next_image.extend_from_slice(pixel(image, width, row, column));
            next_indices.push(source_indices[row * width + column]);
        }
    }

    (next_image, next_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_multiple_vertical_seams_and_tracks_source_coordinates() {
        let image: Vec<u8> = (0..4 * 6 * CHANNELS).map(|value| value as u8).collect();
        let original = image.clone();

        let plan = plan_vertical_gradient(&image, 4, 6, 3).unwrap();

        assert_eq!(plan.result.len(), 4 * 3 * CHANNELS);
        assert_eq!(
            plan.removed_mask.iter().filter(|removed| **removed).count(),
            12
        );
        for row in 0..4 {
            let start = row * 6;
            let end = start + 6;
            assert_eq!(
                plan.removed_mask[start..end]
                    .iter()
                    .filter(|removed| **removed)
                    .count(),
                3
            );
        }
        assert_eq!(image, original);
    }

    #[test]
    fn one_seam_is_connected() {
        let image = vec![0; 4 * 5 * CHANNELS];
        let energy = gradient_energy(&image, 4, 5);
        let seam = find_seam(&energy, 4, 5).unwrap();

        assert_eq!(seam.len(), 4);
        assert!(seam.windows(2).all(|pair| pair[0].abs_diff(pair[1]) <= 1));
    }

    #[test]
    fn unchanged_width_returns_copy_and_empty_mask() {
        let image = vec![7; 2 * 3 * CHANNELS];
        let plan = plan_vertical_gradient(&image, 2, 3, 3).unwrap();

        assert_eq!(plan.result, image);
        assert!(plan.removed_mask.iter().all(|removed| !removed));
    }

    #[test]
    fn rejects_invalid_dimensions_and_buffer_lengths() {
        assert_eq!(
            plan_vertical_gradient(&[], 0, 2, 1),
            Err(EngineError::EmptyImage)
        );
        assert_eq!(
            plan_vertical_gradient(&[0; 3], 1, 2, 1),
            Err(EngineError::InvalidImageLength {
                expected: 6,
                actual: 3,
            })
        );
        assert_eq!(
            plan_vertical_gradient(&[0; 6], 1, 2, 0),
            Err(EngineError::InvalidTargetWidth {
                target: 0,
                source: 2,
            })
        );
        assert_eq!(
            plan_vertical_gradient(&[0; 6], 1, 2, 3),
            Err(EngineError::InvalidTargetWidth {
                target: 3,
                source: 2,
            })
        );
    }
}
