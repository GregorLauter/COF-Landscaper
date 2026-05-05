#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INPUT_FILE="$SCRIPT_DIR/USAGE.md"
OUTPUT_FILE="$SCRIPT_DIR/USAGE.pdf"
PDF_ENGINE="${PDF_ENGINE:-tectonic}"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "error: input file not found: $INPUT_FILE" >&2
  exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "error: pandoc is required to build USAGE.pdf" >&2
  echo "install: https://pandoc.org/installing.html" >&2
  exit 1
fi

if ! command -v "$PDF_ENGINE" >/dev/null 2>&1; then
  echo "error: PDF engine '$PDF_ENGINE' not found" >&2
  echo "hint: install tectonic or set PDF_ENGINE to a working engine" >&2
  exit 1
fi

pandoc "$INPUT_FILE" \
  --standalone \
  --pdf-engine="$PDF_ENGINE" \
  --from "markdown+yaml_metadata_block+implicit_figures+table_captions+fenced_code_attributes" \
  --toc \
  --toc-depth=3 \
  --number-sections \
  --highlight-style=tango \
  -V geometry:margin=2cm \
  -V fontsize=11pt \
  -V linestretch=1.08 \
  -V colorlinks=true \
  --resource-path="$SCRIPT_DIR:$SCRIPT_DIR/figures:$SCRIPT_DIR/../.." \
  -o "$OUTPUT_FILE"

echo "Wrote $OUTPUT_FILE"
