use crate::image::pixel;
use crate::CHANNELS;

pub(crate) fn compact(
    image: &[u8],
    source_indices: &[usize],
    height: usize,
    width: usize,
    seam: &[usize],
    next_image: &mut Vec<u8>,
    next_indices: &mut Vec<usize>,
) {
    next_image.clear();
    next_indices.clear();

    for row in 0..height {
        for column in 0..width {
            if column == seam[row] {
                continue;
            }

            next_image.extend_from_slice(pixel(image, width, row, column));
            next_indices.push(source_indices[row * width + column]);
        }
    }
}

pub(crate) fn transpose_pixels(
    image: &[u8],
    height: usize,
    width: usize,
) -> Vec<u8> {
    let mut transposed = vec![0; image.len()];

    for row in 0..height {
        for column in 0..width {
            let source_start = (row * width + column) * CHANNELS;
            let target_start = (column * height + row) * CHANNELS;
            transposed[target_start..target_start + CHANNELS]
                .copy_from_slice(&image[source_start..source_start + CHANNELS]);
        }
    }

    transposed
}

pub(crate) fn transpose_indices(
    indices: &[usize],
    height: usize,
    width: usize,
) -> Vec<usize> {
    let mut transposed = vec![0; indices.len()];

    for row in 0..height {
        for column in 0..width {
            transposed[column * height + row] = indices[row * width + column];
        }
    }

    transposed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_replaces_existing_destination_contents() {
        let image: Vec<u8> = (0..18).collect();
        let source_indices = vec![10, 11, 12, 13, 14, 15];
        let seam = vec![1, 0];
        let mut next_image = vec![255; 3];
        let mut next_indices = vec![99];

        compact(
            &image,
            &source_indices,
            2,
            3,
            &seam,
            &mut next_image,
            &mut next_indices,
        );

        assert_eq!(next_image, vec![0, 1, 2, 6, 7, 8, 12, 13, 14, 15, 16, 17]);
        assert_eq!(next_indices, vec![10, 12, 14, 15]);
    }
}
