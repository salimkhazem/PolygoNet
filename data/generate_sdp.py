import os
import sys
import time

# Local import
from thresholding import threshold_img_to_sdp


def generate_sdp(input_folder, dataset, output_folder=None):
    """
    input_folder: input folder which contains images
    output_folder: output folder where we want to save sdp files
    dataset: dataset name to save the process time 
    """
    if output_folder is None:
        output_folder = f"{input_folder}_sdp"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = [
        f
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
    ]
    process_time = 0
    for image_file in image_files:
        try:
            input_image_path = os.path.join(input_folder, image_file)
            output_sdp_file = os.path.join(
                output_folder, f"{os.path.splitext(image_file)[0]}.sdp"
            )
            start = time.time()
            process = threshold_img_to_sdp(input_image_path, output_sdp_file)
            end = time.time()
            process_time = (end - start) * 1000
            print(f"Process time {process} ms")
        except Exception as e:
            print(f"Error: {e}")

    with open(f"{dataset}", "a") as file:
        file.write(f"\nContour and SDP generation time: {process:.2f} (ms)")


if __name__ == "__main__":
    input = sys.argv[1]
    dataset = sys.argv[2]
    generate_sdp(input, dataset=dataset)
