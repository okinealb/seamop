import numpy as np

from seamop._plan import build_plan


class LeftEdgeCalculator:
    def __init__(self):
        self.calls = []

    def __call__(self, image, num_seams):
        self.calls.append((image.shape, num_seams))
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[:, :num_seams] = True
        return mask


def test_preview_marks_exactly_the_pixels_missing_from_result():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    original = image.copy()
    calculator = LeftEdgeCalculator()

    plan = build_plan(image, height=3, width=3, seam_finder=calculator)
    result = plan.result()
    preview = plan.preview((255, 255, 255))

    removed = np.zeros(image.shape[:2], dtype=bool)
    removed[:, :2] = True
    removed[0, 2:] = True

    assert plan.source_shape == (4, 5, 3)
    assert plan.target_shape == (3, 3, 3)
    assert calculator.calls == [((4, 5, 3), 2), ((3, 4, 3), 1)]
    assert np.array_equal(result, original[1:, 2:])
    assert np.all(preview[removed] == 255)
    assert np.array_equal(preview[~removed], original[~removed])
    assert np.array_equal(image, original)


def test_plan_owns_state_and_returns_independent_images():
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    expected_source = image.copy()
    plan = build_plan(
        image,
        height=2,
        width=3,
        seam_finder=LeftEdgeCalculator(),
    )
    expected_result = expected_source[1:, 1:]

    image.fill(0)
    result = plan.result()
    preview = plan.preview((255, 255, 255))
    result.fill(0)
    preview.fill(0)
    fresh_result = plan.result()
    fresh_preview = plan.preview((255, 255, 255))

    assert np.array_equal(fresh_result, expected_result)
    assert np.array_equal(
        fresh_preview[1:, 1:],
        expected_source[1:, 1:],
    )
    assert not np.shares_memory(result, fresh_result)
    assert not np.shares_memory(preview, fresh_preview)
