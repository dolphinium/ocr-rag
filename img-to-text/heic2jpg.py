import os
from pillow_heif import open_heif
from PIL import Image

# Define input and output folder
input_folder = "../images_heic_2"
output_folder = "../images_jpg_2"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a sorted list of HEIC files to maintain order
heic_files = sorted(
    [f for f in os.listdir(input_folder) if f.lower().endswith(".heic")]
)

# Convert and rename sequentially
for idx, file in enumerate(heic_files, start=1):
    heif_file = open_heif(os.path.join(input_folder, file))
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode
    )
    output_path = os.path.join(output_folder, f"{idx}.jpg")
    image.save(output_path, format="JPEG")
    print(f"Converted: {file} -> {output_path}")

print("Conversion complete!")
