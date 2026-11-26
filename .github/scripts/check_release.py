"""Verify the tag version and packaged topology data before publication."""

# Standalone CI entry point.
# ruff: noqa: INP001

import sys
import tarfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def check_topologies(files: set[str], prefix: str) -> None:
    """Require the five topology definitions and reject generated caches."""
    root = f"{prefix}coflandscaper/database/topologies/"
    required = {
        f"{root}{name}.cgd"
        for name in ("hcb", "hcb_ab", "kgm", "sql", "sql_ab")
    }
    missing = required - files
    caches = {
        name
        for name in files
        if name.startswith(root) and name.endswith(".pickle")
    }
    if missing or caches:
        raise ValueError(
            f"Missing topology files: {sorted(missing)}; unexpected caches: {sorted(caches)}"
        )


def check_version(
    metadata: bytes, filename_version: Version, tag: str
) -> None:
    """Check embedded and filename versions against the normalised release tag."""
    version = BytesParser().parsebytes(metadata)["Version"]
    expected = str(Version(tag.removeprefix("v")))
    if version != expected or str(filename_version) != expected:
        raise ValueError(
            f"{tag} expects {expected}, found metadata {version} and filename {filename_version}"
        )


def main(tag: str, wheel: Path, sdist: Path) -> None:
    """Validate the same wheel and source archive that the workflow uploads."""
    with ZipFile(wheel) as archive:
        files = set(archive.namelist())
        (metadata,) = (
            name for name in files if name.endswith(".dist-info/METADATA")
        )
        check_version(
            archive.read(metadata), parse_wheel_filename(wheel.name)[1], tag
        )
        check_topologies(files, "")

    root = sdist.name.removesuffix(".tar.gz")
    with tarfile.open(sdist, "r:gz") as source:
        metadata_file = source.extractfile(f"{root}/PKG-INFO")
        if metadata_file is None:
            raise ValueError("Missing source distribution metadata")
        check_version(
            metadata_file.read(), parse_sdist_filename(sdist.name)[1], tag
        )
        check_topologies(set(source.getnames()), f"{root}/src/")


if __name__ == "__main__":
    release_tag, wheel_path, sdist_path = sys.argv[1:]
    main(release_tag, Path(wheel_path), Path(sdist_path))
