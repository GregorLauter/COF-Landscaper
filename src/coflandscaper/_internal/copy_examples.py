"""CLI for downloading and copying COF-Landscaper examples."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

ARCHIVE_URL = "https://github.com/GregorLauter/COF-Landscaper/archive/refs/heads/master.zip"
ARCHIVE_ROOT = "COF-Landscaper-master"
EXAMPLES_DIR = "examples"


def _download_archive(url: str, archive_path: Path) -> None:
    """Download an HTTPS archive URL to a local path."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SystemExit(f"Refusing to download non-HTTPS URL: {url}")

    try:
        with (
            urllib.request.urlopen(url) as response,  # noqa: S310
            archive_path.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
    except urllib.error.URLError as exc:
        raise SystemExit("Failed to download the GitHub archive.") from exc
    except OSError as exc:
        raise SystemExit("Failed to write downloaded GitHub archive.") from exc


def _copy_examples_from_archive(
    archive_path: Path, output_dir: Path, force: bool
) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            prefix = f"{ARCHIVE_ROOT}/{EXAMPLES_DIR}/"
            members = [
                name
                for name in archive.namelist()
                if name.startswith(prefix) and not name.endswith("/")
            ]
            if not members:
                raise SystemExit(
                    "No examples/ directory found in the GitHub archive."
                )

            output_dir.mkdir(parents=True, exist_ok=True)

            for member in members:
                rel_path = Path(member[len(prefix) :])
                dest_path = output_dir / EXAMPLES_DIR / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if dest_path.exists() and not force:
                    raise SystemExit(
                        f"File already exists: {dest_path}. Use --force to overwrite."
                    )
                with archive.open(member) as src, dest_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(dest_path)
    except zipfile.BadZipFile as exc:
        raise SystemExit("Failed to read the GitHub archive.") from exc
    except OSError as exc:
        raise SystemExit("Failed to write example files.") from exc


def copy_examples(output_dir: Path, force: bool) -> None:
    """Download the GitHub archive and copy examples into output_dir."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "cof-landscaper.zip"
            _download_archive(ARCHIVE_URL, archive_path)
            _copy_examples_from_archive(archive_path, output_dir, force)
    except urllib.error.URLError as exc:
        raise SystemExit("Failed to download the GitHub archive.") from exc
    except RuntimeError as exc:
        raise SystemExit(
            "Unexpected error while downloading examples."
        ) from exc


def main() -> None:
    """Entrypoint for the copy-examples CLI."""
    parser = argparse.ArgumentParser(
        description="Download COF-Landscaper examples into a directory.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Destination directory (defaults to current working directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    args = parser.parse_args()

    copy_examples(Path(args.output_dir), args.force)


if __name__ == "__main__":
    main()
