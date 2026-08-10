"""Command-line image resizing, removal, and highlighting."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from cyclopts import App, CycloptsError, Parameter
from cyclopts.help import ColumnSpec, DefaultFormatter, DescriptionRenderer, HelpEntry
from PIL import Image

from ._image import ImageDecodeError, normalize_image
from ._plan import DEFAULT_HIGHLIGHT_COLOR
from ._validation import validate_num_seams
from .core import CarvingStrategy, plan, resize
from .logger import setup_cli_logging
from .methods import EnergyMethod, GradientEnergy, LaplacianEnergy, SobelEnergy

EnergyName = Literal["gradient", "sobel", "laplacian"]
StrategyName = Literal["backward", "forward"]
Direction = Literal["vertical", "horizontal"]
CommandName = Literal["resize", "remove", "highlight"]

_ENERGY_METHODS: dict[str, type[EnergyMethod]] = {
    "gradient": GradientEnergy,
    "sobel": SobelEnergy,
    "laplacian": LaplacianEnergy,
}


def _short_names(entry: HelpEntry) -> str:
    return " ".join(entry.positive_shorts)


def _long_names(entry: HelpEntry) -> str:
    return " ".join(entry.positive_names)


_HELP_COLUMNS = (
    ColumnSpec(
        renderer=lambda entry: "*" if entry.required else "",
        width=1,
        style="red bold",
    ),
    ColumnSpec(renderer=_short_names, width=2, style="cyan"),
    ColumnSpec(renderer=_long_names, no_wrap=True, style="cyan"),
    ColumnSpec(renderer=DescriptionRenderer(), overflow="fold"),
)


def _format_usage_error(error: CycloptsError) -> str:
    command = " ".join(("seamop", *(error.command_chain or ()), "--help"))
    return f"Error: {error}\nTry '{command}' for more information."


app = App(
    name="seamop",
    help="A command-line tool for seam carving images.",
    default_parameter=Parameter(negative=False),
    help_formatter=DefaultFormatter(column_specs=_HELP_COLUMNS),
    help_on_error=False,
    error_formatter=_format_usage_error,
    version_flags=(),
    result_action="return_value",
)


@Parameter(name="*")
@dataclass(frozen=True)
class CommonOptions:
    """Options shared by all image operations.

    Parameters
    ----------
    output
        Output path. A descriptive name is used when omitted.
    energy
        Energy method used to rank pixels with the backward strategy. If
        omitted, forward strategy uses pure forward energy.
    strategy
        Seam-carving strategy.
    log_file
        Path to save the log file.
    verbose
        Enable verbose debugging output.
    quiet
        Suppress output except warnings and errors.
    """

    output: Annotated[Path | None, Parameter(alias="-o")] = None
    energy: Annotated[EnergyName | None, Parameter(alias="-e")] = None
    strategy: StrategyName = "backward"
    log_file: Annotated[Path | None, Parameter(alias="-l")] = None
    verbose: Annotated[bool, Parameter(alias="-v")] = False
    quiet: Annotated[bool, Parameter(alias="-q")] = False


@app.command(name="resize", sort_key=0)
def resize_command(
    input: Path,
    width: int,
    height: int,
    *,
    options: CommonOptions = CommonOptions(),
) -> None:
    """Resize an image by removing seams.

    Parameters
    ----------
    input
        Path to the input image file.
    width
        Output width.
    height
        Output height.
    """
    _execute(
        "resize",
        input,
        options,
        height=height,
        width=width,
    )


@app.command(name="remove", sort_key=1)
def remove_command(
    input: Path,
    *,
    direction: Annotated[Direction, Parameter(alias="-d")] = "vertical",
    count: Annotated[int, Parameter(alias="-c")] = 1,
    options: CommonOptions = CommonOptions(),
) -> None:
    """Remove seams from an image.

    Parameters
    ----------
    input
        Path to the input image file.
    direction
        Direction of seams to remove.
    count
        Number of seams to remove.
    """
    _execute(
        "remove",
        input,
        options,
        direction=direction,
        count=count,
    )


@app.command(name="highlight", sort_key=2)
def highlight_command(
    input: Path,
    width: int,
    height: int,
    *,
    rgb: Annotated[
        tuple[int, int, int],
        Parameter(alias="-r"),
    ] = DEFAULT_HIGHLIGHT_COLOR,
    options: CommonOptions = CommonOptions(),
) -> None:
    """Highlight the pixels removed to reach target dimensions.

    Parameters
    ----------
    input
        Path to the input image file.
    width
        Target width.
    height
        Target height.
    rgb
        RGB highlight color.
    """
    _execute(
        "highlight",
        input,
        options,
        height=height,
        width=width,
        rgb=rgb,
    )


def _execute(
    command: CommandName,
    input: Path,
    options: CommonOptions,
    *,
    height: int | None = None,
    width: int | None = None,
    direction: Direction | None = None,
    count: int | None = None,
    rgb: tuple[int, int, int] = DEFAULT_HIGHLIGHT_COLOR,
) -> None:
    logger = setup_cli_logging(
        verbose=options.verbose,
        quiet=options.quiet,
        log_file=None if options.log_file is None else str(options.log_file),
    )

    try:
        logger.info(f"Loading image from {input}...")
        image = normalize_image(input)
        logger.debug(f"Image loaded with shape {image.shape}.")
        strategy = CarvingStrategy(options.strategy)
        energy: EnergyMethod | None
        if options.energy is None:
            energy = None if strategy is CarvingStrategy.FORWARD else GradientEnergy()
        else:
            energy = _ENERGY_METHODS[options.energy]()

        if command == "remove":
            assert direction is not None and count is not None
            target_height, target_width = image.shape[:2]
            if direction == "vertical":
                count = validate_num_seams(count, target_width)
                target_width -= count
            else:
                count = validate_num_seams(count, target_height)
                target_height -= count
            descriptor = f"removed_{count}_{direction}"
        else:
            assert height is not None and width is not None
            target_height, target_width = height, width
            descriptor = f"{'resized' if command == 'resize' else 'highlighted'}"
            descriptor += f"_{width}x{height}"

        output_path = _get_output_path(input, options.output, descriptor)
        started = perf_counter()

        if command == "resize":
            logger.info(f"Resizing image to {width}x{height}...")
            result = resize(
                image,
                height=target_height,
                width=target_width,
                energy=energy,
                strategy=strategy,
            )
            logger.info("Image resized successfully.")
        else:
            if command == "remove":
                logger.info(f"Removing {count} seams in {direction} direction...")
            else:
                logger.info(f"Highlighting removals for resize to {width}x{height}...")

            resize_plan = plan(
                image,
                height=target_height,
                width=target_width,
                energy=energy,
                strategy=strategy,
            )
            if command == "remove":
                result = resize_plan.result()
                logger.info("Seams removed successfully.")
            else:
                result = resize_plan.preview(rgb)
                logger.info("Seams highlighted successfully.")

        elapsed = perf_counter() - started
        logger.info(f"Processing completed in {elapsed:.3f} seconds.")
        logger.info(f"Saving output image to {output_path}...")
        Image.fromarray(result).save(output_path)
        logger.info(f"Output image saved to {output_path}.")
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user.")
        raise SystemExit(130) from None
    except Exception as error:
        handle_error(error, logger, verbose=options.verbose)
        raise SystemExit(1) from None


def handle_error(
    error: Exception,
    logger: logging.Logger,
    verbose: bool = False,
) -> None:
    """Write an actionable message for an image-processing failure."""
    if isinstance(error, FileExistsError):
        logger.error(str(error))
        logger.error("Choose another output path and try again.")
    elif isinstance(error, FileNotFoundError):
        logger.error(f"File not found: {error.filename}")
        logger.error("Please check the file path and try again.")
    elif isinstance(error, PermissionError):
        logger.error(f"Permission denied: {error.filename}")
        logger.error(
            "Please check file permissions or run the command with elevated privileges."
        )
    elif isinstance(error, ImageDecodeError):
        logger.error("Invalid image file format.")
        logger.error("Use one of the PIL supported formats: PNG, JPEG, BMP, etc.")
    elif isinstance(error, ValueError):
        logger.error(f"Invalid input: {error}")
    elif isinstance(error, MemoryError):
        logger.error("Not enough memory to process the image.")
        logger.error("Try using a smaller image or increasing available memory.")
    else:
        logger.error("An unexpected error occurred.")
        if not verbose:
            logger.error("Use -v/--verbose for more details.")

    if verbose:
        logger.debug("Error details:", exc_info=error)


def _get_output_path(
    input: Path,
    output: Path | None,
    descriptor: str,
) -> Path:
    """Return an unused explicit or derived output path."""
    if output is None:
        suffix = input.suffix or ".png"
        output = Path.cwd() / f"{input.stem}_{descriptor}{suffix}"

    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    app(argv)


if __name__ == "__main__":
    main()
