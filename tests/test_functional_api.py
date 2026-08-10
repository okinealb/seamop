from pathlib import Path

import numpy as np
import pytest

from seamop import CarvingStrategy, ResizePlan, plan, resize


class FailingEnergy:
    def __init__(self, fail_on_call):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, image):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("energy calculation failed")
        return np.zeros(image.shape[:2], dtype=np.float32)


class TestResize:
    def test_returns_owned_image_without_mutating_input(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        original = image.copy()

        result = resize(image, height=3, width=3)

        assert result.shape == (3, 3, 3)
        assert result.dtype == np.uint8
        assert result.flags.owndata
        assert not np.shares_memory(result, image)
        assert np.array_equal(image, original)

    def test_matches_planned_result(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

        result = resize(image, height=3, width=3)
        resize_plan = plan(image, height=3, width=3)

        assert np.array_equal(result, resize_plan.result())

    def test_matches_width_first_sequential_resize(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

        result = resize(image, height=3, width=3)
        width_first = resize(image, height=4, width=3)
        width_first = resize(width_first, height=3, width=3)

        assert np.array_equal(result, width_first)

    def test_same_size_returns_independent_image(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)

        result = resize(image, height=2, width=3)

        assert np.array_equal(result, image)
        assert not np.shares_memory(result, image)

    def test_accepts_pathlike_input(self, input_image_path):
        result = resize(Path(input_image_path), height=4, width=5)

        assert result.shape == (4, 5, 3)

    def test_targets_are_keyword_only(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)

        with pytest.raises(TypeError):
            resize(image, 2, 2)

    @pytest.mark.parametrize(
        ("height", "width", "exception"),
        [
            (0, 2, ValueError),
            (2, 0, ValueError),
            (4, 2, ValueError),
            (2, 4, ValueError),
            (2.0, 2, TypeError),
            (2, True, TypeError),
        ],
        ids=[
            "zero-height",
            "zero-width",
            "larger-height",
            "larger-width",
            "float-height",
            "bool-width",
        ],
    )
    def test_rejects_invalid_target(self, height, width, exception):
        image = np.zeros((3, 3, 3), dtype=np.uint8)

        with pytest.raises(exception):
            resize(image, height=height, width=width)

    def test_failure_does_not_mutate_input(self):
        image = np.arange(3 * 3 * 3, dtype=np.uint8).reshape(3, 3, 3)
        original = image.copy()

        with pytest.raises(RuntimeError, match="energy calculation failed"):
            resize(
                image,
                height=2,
                width=2,
                energy=FailingEnergy(fail_on_call=2),
            )

        assert np.array_equal(image, original)

    def test_forward_strategy_returns_target_without_mutating_input(self):
        image = np.random.default_rng(1).integers(
            0,
            256,
            (6, 8, 3),
            dtype=np.uint8,
        )
        original = image.copy()

        result = resize(
            image,
            height=4,
            width=6,
            strategy=CarvingStrategy.FORWARD,
        )

        assert result.shape == (4, 6, 3)
        assert result.dtype == np.uint8
        assert np.array_equal(image, original)

    def test_forward_strategy_rejects_energy(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="does not accept"):
            resize(
                image,
                height=3,
                width=3,
                energy=lambda current: np.zeros(current.shape[:2]),
                strategy=CarvingStrategy.FORWARD,
            )

    @pytest.mark.parametrize("operation", [resize, plan])
    def test_method_keyword_is_removed(self, operation):
        image = np.zeros((3, 4, 3), dtype=np.uint8)

        with pytest.raises(TypeError):
            operation(
                image,
                height=3,
                width=3,
                method=lambda current: np.zeros(current.shape[:2]),
            )

    def test_rejects_string_strategy(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)

        with pytest.raises(TypeError, match="CarvingStrategy"):
            resize(image, height=3, width=3, strategy="forward")


class TestPlan:
    def test_reuses_computed_seams(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        calls = 0

        def left_edge_energy(current):
            nonlocal calls
            calls += 1
            columns = np.arange(current.shape[1], dtype=np.float32)
            return np.broadcast_to(columns, current.shape[:2]).copy()

        resize_plan = plan(image, height=3, width=2, energy=left_edge_energy)
        first_result = resize_plan.result()
        second_result = resize_plan.result()
        first_preview = resize_plan.preview()
        second_preview = resize_plan.preview()

        assert isinstance(resize_plan, ResizePlan)
        assert repr(resize_plan) == (
            "ResizePlan(source_shape=(3, 4, 3), target_shape=(3, 2, 3))"
        )
        assert resize_plan.source_shape == (3, 4, 3)
        assert resize_plan.target_shape == (3, 2, 3)
        assert calls == 2
        assert np.array_equal(first_result, second_result)
        assert np.array_equal(first_preview, second_preview)
        assert not np.shares_memory(first_result, second_result)
        assert not np.shares_memory(first_preview, second_preview)

    @pytest.mark.parametrize(
        ("color", "exception"),
        [
            ((1, 2), ValueError),
            ((1, 2, 3, 4), ValueError),
            ((-1, 2, 3), ValueError),
            ((1, 2, 256), ValueError),
            ((1, 2, 3.0), TypeError),
            ((1, 2, True), TypeError),
        ],
        ids=[
            "two-channels",
            "four-channels",
            "negative",
            "over-255",
            "float",
            "boolean",
        ],
    )
    def test_preview_rejects_invalid_color(self, color, exception):
        resize_plan = plan(
            np.zeros((2, 3, 3), dtype=np.uint8),
            height=2,
            width=2,
        )

        with pytest.raises(exception):
            resize_plan.preview(color)
