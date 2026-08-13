use crate::CHANNELS;

const BORDER_ENERGY: f32 = 1000.0;

pub(crate) fn gradient_energy(image: &[u8], height: usize, width: usize) -> Vec<f32> {
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
