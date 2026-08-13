//! End-to-end planning for the built-in carving strategies.
//!
//! Width reduction runs first. If height must also shrink, the working image
//! and source-index map are transposed so the same vertical seam machinery can
//! remove horizontal seams. The engine keeps the source coordinates throughout
//! both stages and returns the final image plus one source-sized mask.

use std::error::Error;
use std::fmt;

use crate::compact::compact;
use crate::energy::gradient_energy;
use crate::CHANNELS;
use crate::{forward, seam, transpose};

#[derive(Clone, Copy)]
enum Strategy {
    Gradient,
    Forward,
}

struct Workspace {
    image: Vec<u8>,
    source_indices: Vec<usize>,
    next_image: Vec<u8>,
    next_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised while validating or executing an engine resize plan.
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
            Self::EmptyImage => {
                formatter.write_str("image dimensions must be positive")
            }
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
            Self::SizeOverflow => {
                formatter.write_str("image dimensions overflow the buffer size")
            }
        }
    }
}

impl Error for EngineError {}

#[derive(Debug, PartialEq, Eq)]
/// The result of one complete built-in resize plan.
pub struct Plan {
    /// Final image pixels in row-major RGB order.
    pub result: Vec<u8>,
    /// Pixels removed from the original image, in row-major order.
    pub removed_mask: Vec<bool>,
    /// Height of the source image.
    pub source_height: usize,
    /// Width of the source image.
    pub source_width: usize,
    /// Height of the result image.
    pub target_height: usize,
    /// Width of the result image.
    pub target_width: usize,
}

/// Plan a resize with backward gradient energy.
pub fn plan(
    image: &[u8],
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
) -> Result<Plan, EngineError> {
    plan_with_strategy(
        image,
        height,
        width,
        target_height,
        target_width,
        Strategy::Gradient,
    )
}

/// Plan a resize with forward transition costs.
pub fn plan_forward(
    image: &[u8],
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
) -> Result<Plan, EngineError> {
    plan_with_strategy(
        image,
        height,
        width,
        target_height,
        target_width,
        Strategy::Forward,
    )
}

fn plan_with_strategy(
    image: &[u8],
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
    strategy: Strategy,
) -> Result<Plan, EngineError> {
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

    let mut workspace = Workspace {
        image: image.to_vec(),
        source_indices: (0..pixel_count).collect(),
        next_image: Vec::with_capacity(expected_bytes),
        next_indices: Vec::with_capacity(pixel_count),
    };
    let mut removed_mask = vec![false; pixel_count];
    let mut current_width = width;

    remove_seams(
        &mut workspace,
        height,
        &mut current_width,
        target_width,
        &mut removed_mask,
        strategy,
    )?;

    if target_height < height {
        let oriented_height = current_width;
        workspace.image = transpose::transpose_pixels(
            &workspace.image,
            height,
            current_width,
        );
        workspace.source_indices = transpose::transpose_indices(
            &workspace.source_indices,
            height,
            current_width,
        );

        let mut oriented_width = height;
        remove_seams(
            &mut workspace,
            oriented_height,
            &mut oriented_width,
            target_height,
            &mut removed_mask,
            strategy,
        )?;

        workspace.image = transpose::transpose_pixels(
            &workspace.image,
            oriented_height,
            target_height,
        );
    }

    Ok(Plan {
        result: workspace.image,
        removed_mask,
        source_height: height,
        source_width: width,
        target_height,
        target_width,
    })
}

fn remove_seams(
    workspace: &mut Workspace,
    height: usize,
    width: &mut usize,
    target_width: usize,
    removed_mask: &mut [bool],
    strategy: Strategy,
) -> Result<(), EngineError> {
    while *width > target_width {
        let seam = match strategy {
            Strategy::Gradient => {
                let energy = gradient_energy(&workspace.image, height, *width);
                seam::find_seam(&energy, height, *width)?
            }
            Strategy::Forward => {
                forward::find_seam(&workspace.image, height, *width)?
            }
        };

        for (row, &column) in seam.iter().enumerate() {
            let current_index = row * *width + column;
            removed_mask[workspace.source_indices[current_index]] = true;
        }

        compact(
            &workspace.image,
            &workspace.source_indices,
            height,
            *width,
            &seam,
            &mut workspace.next_image,
            &mut workspace.next_indices,
        );
        std::mem::swap(&mut workspace.image, &mut workspace.next_image);
        std::mem::swap(
            &mut workspace.source_indices,
            &mut workspace.next_indices,
        );
        *width -= 1;
    }

    Ok(())
}

fn checked_pixel_count(
    height: usize,
    width: usize,
) -> Result<usize, EngineError> {
    height.checked_mul(width).ok_or(EngineError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_plan_marks_removed_source_pixels() {
        let image: Vec<u8> =
            (0..4 * 6 * CHANNELS).map(|value| value as u8).collect();
        let original = image.clone();

        let plan = plan(&image, 4, 6, 4, 3).unwrap();

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

        let plan = plan(&image, 4, 5, 3, 5).unwrap();

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
        let image: Vec<u8> =
            (0..4 * 5 * CHANNELS).map(|value| value as u8).collect();
        let original = image.clone();

        let plan = plan(&image, 4, 5, 3, 3).unwrap();

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
        let plan = plan(&image, 2, 3, 2, 3).unwrap();

        assert_eq!(plan.result, image);
        assert!(plan.removed_mask.iter().all(|removed| !removed));
    }

    #[test]
    fn plan_forward_handles_vertical_removal() {
        let image: Vec<u8> =
            (0..4 * 5 * CHANNELS).map(|value| value as u8).collect();
        let original = image.clone();

        let plan = plan_forward(&image, 4, 5, 4, 3).unwrap();

        assert_eq!(plan.result.len(), 4 * 3 * CHANNELS);
        assert_eq!(
            plan.removed_mask.iter().filter(|removed| **removed).count(),
            8
        );
        for row in 0..4 {
            assert_eq!(
                plan.removed_mask[row * 5..row * 5 + 5]
                    .iter()
                    .filter(|removed| **removed)
                    .count(),
                2
            );
        }
        assert_eq!(image, original);
    }

    #[test]
    fn plan_forward_handles_horizontal_removal() {
        let image: Vec<u8> =
            (0..4 * 5 * CHANNELS).map(|value| value as u8).collect();
        let original = image.clone();

        let plan = plan_forward(&image, 4, 5, 3, 5).unwrap();

        assert_eq!(plan.result.len(), 3 * 5 * CHANNELS);
        assert_eq!(
            plan.removed_mask.iter().filter(|removed| **removed).count(),
            5
        );
        for column in 0..5 {
            let rows: Vec<usize> = (0..4)
                .filter(|row| plan.removed_mask[row * 5 + column])
                .collect();
            assert_eq!(rows.len(), 1);
        }
        assert_eq!(image, original);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(plan(&[], 0, 2, 1, 1), Err(EngineError::EmptyImage));
        assert_eq!(
            plan(&[0; 3], 1, 2, 1, 1),
            Err(EngineError::InvalidImageLength {
                expected: 6,
                actual: 3,
            })
        );
        assert_eq!(
            plan(&[0; 6], 1, 2, 0, 2),
            Err(EngineError::InvalidTargetHeight {
                target: 0,
                source: 1,
            })
        );
        assert_eq!(
            plan(&[0; 6], 1, 2, 1, 0),
            Err(EngineError::InvalidTargetWidth {
                target: 0,
                source: 2,
            })
        );
        assert_eq!(
            plan(&[0; 6], 1, 2, 1, 3),
            Err(EngineError::InvalidTargetWidth {
                target: 3,
                source: 2,
            })
        );
    }
}
