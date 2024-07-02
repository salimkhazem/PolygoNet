#!/bin/bash

# Set the input folder path
input_folder=$1

# Set the output folder path
output_folder=$2

mkdir $output_folder
# Path to the MATCSimplifiedDominantPoint executable
executable="./MATCSimplifiedDominantPoint"

# Iterate over .sdp files in the input folder
echo "Processing ... "
for file in "$input_folder"/*.sdp; do
    # Extract the file name without extension
    filename=$(basename -- "$file")
    output=$(basename -- "$file" .sdp)

    # Set the output file path
    output_file="$output_folder/$output"

    # Execute the command
    $executable -i "$file" -o "$output_file" -d "../ImaGene"
done
echo "Processing Done"

