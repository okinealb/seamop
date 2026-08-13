use std::error::Error;
use std::fmt;

use crate::compact::{compact, transpose_indices, transpose_pixels};
use crate::energy::gradient_energy;
use crate::seam::find_seam;
use crate::CHANNELS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    EmptyImage,
    InvalidImageLength { expected: usize, actual: usize },
    InvalidTargetHeight { target: usize, source: usize },
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
            Self::InvalidTargetHeight { target, source } => write!(
                formatter,
                "target height must be between 1 and {source}; got {target}"
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
pub struct GradientPlan {
    pub result: Vec<u8>,
    pub removed_mask: Vec<bool>,
    pub source_height: usize,
    pub source_width: usize,
    pub target_height: usize,
    pub target_width: usize,
}

pub fn plan_gradient(
    image: &[u8],
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
) -> Result<GradientPlan, EngineError> {
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
    if target_height == 0 || target_height > height {
        return Err(EngineError::InvalidTargetHeight {
            target: target_height,
            source: height,
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

    remove_seams(
        &mut working,
        &mut source_indices,
        height,
        &mut current_width,
        target_width,
        &mut removed_mask,
    )?;

    if target_height < height {
        let oriented_height = current_width;
        working = transpose_pixels(&working, height, current_width);
        source_indices = transpose_indices(&source_indices, height, current_width);

        let mut oriented_width = height;
        remove_seams(
            &mut working,
            &mut source_indices,
            oriented_height,
            &mut oriented_width,
            target_height,
            &mut removed_mask,
        )?;

        working = transpose_pixels(&working, oriented_height, target_height);
    }

    Ok(GradientPlan {
        result: working,
        removed_mask,
        source_height: height,
        source_width: width,
        target_height,
        target_width,
    })
}

fn remove_seams(
    image: &mut Vec<u8>,
    source_indices: &mut Vec<usize>,
    height: usize,
    width: &mut usize,
    target_width: usize,
    removed_mask: &mut [bool],
) -> Result<(), EngineError> {
    while *width > target_width {
        let energy = gradient_energy(image, height, *width);
        let seam = find_seam(&energy, height, *width)?;

        for row in 0..height {
            let current_index = row * *width + seam[row];
            removed_mask[source_indices[current_index]] = true;
        }

        let (next_image, next_indices) = compact(image, source_indices, height, *width, &seam);
        *image = next_image;
        *source_indices = next_indices;
        *width -= 1;
    }

    Ok(())
}

fn checked_pixel_count(height: usize, width: usize) -> Result<usize, EngineError> {
    height.checked_mul(width).ok_or(EngineError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_plan_marks_removed_source_pixels() {
        let image: Vec<u8> = (0..4 * 6 * CHANNELS).map(|value| value as u8).collect();
        let original = image.clone();

        let plan = plan_gradient(&image, 4, 6, 4, 3).unwrap();

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
    fn horizontal_plan_marks_removed_source_pixels() {
        let image = vec![0; 4 * 5 * CHANNELS];

        let plan = plan_gradient(&image, 4, 5, 3, 5).unwrap();

        assert_eq!(plan.result.len(), 3 * 5 * CHANNELS);
        assert_eq!(
            plan.removed_mask.iter().filter(|removed| **removed).count(),
            5
        );

        let seam_rows: Vec<usize> = (0..5)
            .map(|column| {
                let rows: Vec<usize> = (0..4)
                    .filter(|row| plan.removed_mask[row * 5 + column])
                    .collect();
                assert_eq!(rows.len(), 1);
                rows[0]
            })
            .collect();
        assert!(seam_rows
            .windows(2)
            .all(|pair| pair[0].abs_diff(pair[1]) <= 1));
    }

    #[test]
    fn mixed_plan_marks_all_removed_pixels() {
        let image: Vec<u8> = (0..4 * 5 * CHANNELS).map(|value| value as u8).collect();
        let original = image.clone();

        let plan = plan_gradient(&image, 4, 5, 3, 3).unwrap();

        assert_eq!(plan.result.len(), 3 * 3 * CHANNELS);
        assert_eq!(
            plan.removed_mask.iter().filter(|removed| **removed).count(),
            11
        );
        assert_eq!(image, original);
    }

    #[test]
    fn unchanged_plan_returns_image_and_empty_mask() {
        let image = vec![7; 2 * 3 * CHANNELS];
        let plan = plan_gradient(&image, 2, 3, 2, 3).unwrap();

        assert_eq!(plan.result, image);
        assert!(plan.removed_mask.iter().all(|removed| !removed));
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(plan_gradient(&[], 0, 2, 1, 1), Err(EngineError::EmptyImage));
        assert_eq!(
            plan_gradient(&[0; 3], 1, 2, 1, 1),
            Err(EngineError::InvalidImageLength {
                expected: 6,
                actual: 3,
            })
        );
        assert_eq!(
            plan_gradient(&[0; 6], 1, 2, 0, 2),
            Err(EngineError::InvalidTargetHeight {
                target: 0,
                source: 1,
            })
        );
        assert_eq!(
            plan_gradient(&[0; 6], 1, 2, 1, 0),
            Err(EngineError::InvalidTargetWidth {
                target: 0,
                source: 2,
            })
        );
        assert_eq!(
            plan_gradient(&[0; 6], 1, 2, 1, 3),
            Err(EngineError::InvalidTargetWidth {
                target: 3,
                source: 2,
            })
        );
    }
}
