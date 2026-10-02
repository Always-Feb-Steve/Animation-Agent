#!/usr/bin/env bash

# Usage: bash run_unirig.sh /path/to/filename.glb

if [ "$#" -lt 1 ]; then
    echo "Error: input_path (.glb) is required"
    echo "Usage: bash run_unirig.sh /path/to/filename.glb"
    exit 1
fi

input_path="$1"

# Derive output paths from input_path
input_dir="$(dirname "$input_path")"
input_file="$(basename "$input_path")"
name_no_ext="${input_file%.*}"
temp_path="${input_dir}/${name_no_ext}_skeleton.fbx"
output_path="${input_dir}/${name_no_ext}_skin.fbx"
echo "temp_path: ${temp_path}"
echo "output_path: ${output_path}"

cd UniRig

bash launch/inference/generate_skeleton.sh --input "${input_path}" --output "${temp_path}"
bash launch/inference/generate_skin.sh --input "${temp_path}" --output "${output_path}"

cd ../
