#!/bin/bash

# Define the base directory
BASE_DIR="../data/lifting"

# Iterate over all directories inside the lifting folder
for dir in "$BASE_DIR"/*/; do
    # Extract the folder name
    folder_name=$(basename "$dir")
    
    # Construct the full data path
    DATA_ROOT_PATH="$BASE_DIR/$folder_name/"
    OUTPUT_PATH="$BASE_DIR/$folder_name/"

    # Run the command
    python scripts/data_preprocessing/delta_wrapper.py --ckpt checkpoints/densetrack3d.pth --data_root_path "$DATA_ROOT_PATH" --output_path "$OUTPUT_PATH"