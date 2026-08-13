use crate::CHANNELS;

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
    fn transpose_pixels_reorders_rows_and_columns() {
        let image = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        assert_eq!(
            transpose_pixels(&image, 2, 2),
            vec![1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]
        );
    }

    #[test]
    fn transpose_indices_reorders_rows_and_columns() {
        assert_eq!(
            transpose_indices(&[0, 1, 2, 3, 4, 5], 2, 3),
            vec![0, 3, 1, 4, 2, 5]
        );
    }
}
