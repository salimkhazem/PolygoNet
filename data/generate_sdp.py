import os
import sys
from tqdm import tqdm

# Local import
from thresholding import threshold_img_to_sdp


def generate_sdp(input_folder, output_folder=None):
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
    for image_file in tqdm(image_files, total=len(image_files)):
        try:
            input_image_path = os.path.join(input_folder, image_file)
            output_sdp_file = os.path.join(
                output_folder, f"{os.path.splitext(image_file)[0]}.sdp"
            )
            process = threshold_img_to_sdp(input_image_path, output_sdp_file)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    generate_sdp(input, output)
