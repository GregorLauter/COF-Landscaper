"""Build a user ZIP from the release checkout using Python 3.12+, git and uv.

After the matching tag is on PyPI: python3.12 packaging/user_bundle/build.py
Before publication: append --wheel /path/to/cof_landscaper-....whl
Attach the production ZIP from dist/ to the matching GitHub release.
Local wheel previews are written to the preview/ subfolder.
"""

# Standalone maintainer script invoking git and uv from the maintainer's PATH.
# ruff: noqa: INP001, S603, S607

import argparse
import io
import json
import re
import shutil
import subprocess
import tempfile
import tomllib
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE_NAME = "COF-Landscaper"


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _package_version(wheel: Path | None) -> str:
    if wheel is not None:
        with ZipFile(wheel) as archive:
            (metadata,) = (
                n
                for n in archive.namelist()
                if n.endswith(".dist-info/METADATA")
            )
            info = BytesParser().parsebytes(archive.read(metadata))
        if info["Name"] != "cof-landscaper":
            raise ValueError("Expected a COF-Landscaper wheel.")
        return info["Version"]

    tags = _git("tag", "--points-at", "HEAD").decode().splitlines()
    versions = [
        tag[1:] for tag in tags if re.fullmatch(r"v\d+(?:\.\d+){3}", tag)
    ]
    if len(versions) != 1:
        raise ValueError(
            "Check out a published release tag, or use --wheel for a local test."
        )
    return versions[0]


def _prepare_environment(
    folder: Path, version: str, wheel: Path | None
) -> None:
    config = (
        (HERE / "pyproject.toml.in")
        .read_text(encoding="utf-8")
        .replace("@VERSION@", version)
    )
    if wheel is not None:
        dest = folder / ".test-wheel" / wheel.name
        dest.parent.mkdir()
        shutil.copyfile(wheel, dest)
        relative_path = dest.relative_to(folder).as_posix()
        config += f"\n[tool.uv.sources]\ncof-landscaper = {{ path = {json.dumps(relative_path)} }}\n"
    (folder / "pyproject.toml").write_text(
        config, encoding="utf-8", newline="\n"
    )

    (folder / ".python-version").write_bytes(b"3.12\n")

    # Seed uv's version preferences from the release; resolve only user dependencies.
    lock = folder / "uv.lock"
    seed = _git("show", "HEAD:uv.lock")
    lock.write_bytes(seed)
    subprocess.run(
        ["uv", "lock", "--default-index", "https://pypi.org/simple"],
        cwd=folder,
        check=True,
    )
    pins = {
        (p["name"], p["version"])
        for p in tomllib.loads(seed.decode())["package"]
        if "registry" in p["source"]
    }
    for p in tomllib.loads(lock.read_text(encoding="utf-8"))["package"]:
        if p["name"] == "cof-landscaper":
            if (
                not {"notebook", "porosity"}
                <= p.get("optional-dependencies", {}).keys()
            ):
                raise ValueError(
                    "Expected the release with notebook and porosity extras."
                )
            if wheel is None and p["source"] != {
                "registry": "https://pypi.org/simple"
            }:
                raise ValueError(
                    "Release bundles require COF-Landscaper from PyPI."
                )
        elif (
            "registry" in p["source"] and (p["name"], p["version"]) not in pins
        ):
            raise ValueError(
                f"User dependency differs from the release lock: {p['name']} {p['version']}"
            )


def main() -> None:
    """Create a release or local-wheel preview archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel", type=Path, help="Build a labelled local-wheel test ZIP."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    version = _package_version(args.wheel)
    output_dir = args.output_dir / "preview" if args.wheel else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        folder = Path(work)
        _prepare_environment(folder, version, args.wheel)
        guide = (HERE / "README.md").read_text(encoding="utf-8")
        if args.wheel:
            guide = (
                "**Pre-release installation test.** This ZIP installs the enclosed wheel for COF-Landscaper. Dependencies come from PyPI.\n\n"
                + guide
            )
        tracked = _git(
            "archive", "--format=zip", "HEAD", "examples", "LICENSE"
        )
        # Fixed ZIP timestamps and file modes make repeated builds comparable.
        with (
            ZipFile(io.BytesIO(tracked)) as source,
            ZipFile(folder / "bundle.zip", "w") as out,
        ):

            def add(
                path: str, data: bytes = b"", *, executable: bool = False
            ) -> None:
                entry = ZipInfo(f"{BUNDLE_NAME}/{path}")
                entry.create_system = 3
                if entry.is_dir():
                    entry.external_attr = (0o040755 << 16) | 0x10
                    out.writestr(entry, b"")
                else:
                    mode = 0o100755 if executable else 0o100644
                    entry.external_attr = mode << 16
                    out.writestr(entry, data, compress_type=ZIP_DEFLATED)

            add("")
            for member in source.infolist():
                if member.is_dir():
                    add(member.filename)
                    continue
                data = source.read(member)
                if member.filename.endswith(".ipynb"):
                    notebook = json.loads(data)
                    notebook["metadata"]["kernelspec"].update(
                        name="python3", display_name=BUNDLE_NAME
                    )
                    data = (
                        json.dumps(notebook, indent=1, ensure_ascii=False)
                        + "\n"
                    ).encode()
                # Git's ZIP omits Unix mode bits for ordinary files.
                add(
                    member.filename,
                    data,
                    executable=bool((member.external_attr >> 16) & 0o111),
                )
            add(".vscode/")
            add("README.md", guide.encode("utf-8"))
            for filename in (
                ".vscode/extensions.json",
                ".vscode/settings.json",
            ):
                add(filename, (HERE / filename).read_bytes())
            for filename in ("pyproject.toml", "uv.lock", ".python-version"):
                add(filename, (folder / filename).read_bytes())
            if args.wheel:
                add(".test-wheel/")
                add(f".test-wheel/{args.wheel.name}", args.wheel.read_bytes())
        output = output_dir / f"{BUNDLE_NAME}.zip"
        shutil.copyfile(folder / "bundle.zip", output)
        print(output)


if __name__ == "__main__":
    main()
