use crate::image::pixel;
use crate::CHANNELS;

pub(crate) fn compact(
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
