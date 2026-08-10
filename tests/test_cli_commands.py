import numpy as np
import pytest
from PIL import Image

from seamop.cli import main
from seamop.core import CarvingStrategy
from seamop.methods import GradientEnergy, LaplacianEnergy, SobelEnergy


def test_resize_writes_requested_dimensions(capsys, input_image_path, output_path):
    main(
        [
            "resize",
            "--input",
            input_image_path,
            "--width",
            "5",
            "--height",
            "4",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()

    assert "Loading image" in captured.err
    assert "Resizing image to 5x4" in captured.err
    assert "Saving" in captured.err
    assert str(output_path) in captured.err
    assert "Processing completed in" in captured.err
    with Image.open(output_path) as output:
        assert output.size == (5, 4)


def test_resize_without_output_uses_descriptive_name(
    capsys, input_image_path, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    main(["resize", input_image_path, "5", "4"])

    captured = capsys.readouterr()
    default_output = tmp_path / "input_resized_5x4.png"

    assert "Resizing" in captured.err
    assert str(default_output) in captured.err
    with Image.open(default_output) as output:
        assert output.size == (5, 4)


@pytest.mark.parametrize(
    ("command", "filename"),
    [
        (["remove", "--count", "2"], "input_removed_2_vertical.png"),
        (["highlight", "5", "4"], "input_highlighted_5x4.png"),
    ],
    ids=["remove", "highlight"],
)
def test_command_without_output_uses_descriptive_name(
    command, filename, input_image_path, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "PIL.Image.Image.show",
        lambda self: pytest.fail("CLI attempted to display an image"),
    )

    main([command[0], input_image_path, *command[1:]])

    assert (tmp_path / filename).exists()


@pytest.mark.parametrize(
    ("direction", "count", "size"),
    [
        ("vertical", "2", (5, 6)),
        ("horizontal", "2", (7, 4)),
    ],
    ids=["vertical", "horizontal"],
)
def test_remove_writes_expected_dimensions(
    direction, count, size, capsys, input_image_path, output_path
):
    main(
        [
            "remove",
            input_image_path,
            "-d",
            direction,
            "-c",
            count,
            "-o",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()

    assert "Removing" in captured.err
    assert direction in captured.err
    assert str(output_path) in captured.err
    with Image.open(output_path) as output:
        assert output.size == size


def test_remove_defaults_to_one_seam(capsys, input_image_path, output_path):
    main(
        [
            "remove",
            input_image_path,
            "--direction",
            "vertical",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()

    assert "Removing" in captured.err
    assert str(output_path) in captured.err
    with Image.open(output_path) as output:
        assert output.size == (6, 6)


@pytest.mark.parametrize(
    ("width", "height", "expected_pixels"),
    [
        ("5", "6", 12),
        ("7", "4", 14),
        ("5", "4", 22),
    ],
    ids=["vertical", "horizontal", "both"],
)
def test_highlight_writes_colored_seams(
    width,
    height,
    expected_pixels,
    capsys,
    input_image_path,
    output_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "PIL.Image.Image.show",
        lambda self: pytest.fail("CLI attempted to display an image"),
    )
    args = [
        "highlight",
        input_image_path,
        width,
        height,
        "--output",
        str(output_path),
    ]

    main(args)

    captured = capsys.readouterr()

    assert "Highlighting" in captured.err
    assert f"{width}x{height}" in captured.err
    with Image.open(output_path) as output:
        pixels = np.asarray(output)
        assert output.size == (7, 6)
        assert np.all(pixels == (255, 0, 0), axis=-1).sum() == expected_pixels


def test_highlight_accepts_rgb_alias(input_image_path, output_path):
    main(
        [
            "highlight",
            input_image_path,
            "6",
            "6",
            "-r",
            "1",
            "2",
            "3",
            "-o",
            str(output_path),
        ]
    )

    with Image.open(output_path) as output:
        pixels = np.asarray(output)

    assert np.all(pixels == (1, 2, 3), axis=-1).sum() == 6


@pytest.mark.parametrize(
    ("option", "energy_type"),
    [
        (None, GradientEnergy),
        ("gradient", GradientEnergy),
        ("sobel", SobelEnergy),
        ("laplacian", LaplacianEnergy),
    ],
    ids=["default", "gradient", "sobel", "laplacian"],
)
def test_resize_selects_energy_method(
    option,
    energy_type,
    input_image_path,
    output_path,
    monkeypatch,
):
    selected_energy = None
    selected_strategy = None

    def fake_resize(image, *, height, width, energy, strategy):
        nonlocal selected_energy, selected_strategy
        selected_energy = energy
        selected_strategy = strategy
        return image[:height, :width]

    monkeypatch.setattr("seamop.cli.resize", fake_resize)
    args = ["resize", input_image_path, "5", "4", "--output", str(output_path)]
    if option is not None:
        args.extend(["-e", option])

    main(args)

    assert isinstance(selected_energy, energy_type)
    assert selected_strategy is CarvingStrategy.BACKWARD


def test_resize_selects_forward_strategy_without_energy(
    input_image_path,
    output_path,
):
    main(
        [
            "resize",
            input_image_path,
            "5",
            "4",
            "--strategy",
            "forward",
            "--output",
            str(output_path),
        ]
    )

    with Image.open(output_path) as output:
        assert output.size == (5, 4)
