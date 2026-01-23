#!/bin/bash

# Check if the user has provided an extension as an argument
if [ -z "$1" ]; then
    echo "Please provide a file extension (e.g., .cif, .d12)"
    exit 1
fi

extension="$1"

for file in *"$extension"; do
    if [ ! -e "$file" ]; then
        echo "No files found with extension $extension"
        exit 1
    fi
    base_name="${file%$extension}"
    mkdir -p "$base_name"
    mv "$file" "$base_name/"
    echo "Moved $file to $base_name/"
done