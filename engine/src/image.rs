use crate::CHANNELS;

pub(crate) fn pixel(
    image: &[u8],
    width: usize,
    row: usize,
    column: usize,
) -> &[u8] {
    let start = (row * width + column) * CHANNELS;
    &image[start..start + CHANNELS]
}
