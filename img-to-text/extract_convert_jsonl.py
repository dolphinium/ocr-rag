import os
import json
import easyocr
from tqdm import tqdm

# Define input and output paths
input_folder = "../images_jpg"
output_file = "../book_text.jsonl"  # JSONL format is better for chunking in RAG

# Initialize OCR reader
reader = easyocr.Reader(['tr'])  # Adjust language as needed

# Process images and extract text
book_pages = []

# Sort files to maintain order
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")], key=lambda x: int(x.split('.')[0]))

for img_file in tqdm(image_files, desc="Extracting text"):
    img_path = os.path.join(input_folder, img_file)
    
    # Read text from image
    result = reader.readtext(img_path)
    
    # Extract text content
    text = "\n".join([detection[1] for detection in result])
    
    # Store with page metadata
    book_pages.append({"page": int(img_file.split(".")[0]), "text": text})

# Save the extracted text in JSONL format (one JSON object per line)
with open(output_file, "w", encoding="utf-8") as f:
    for page in book_pages:
        json.dump(page, f, ensure_ascii=False)
        f.write("\n")

print(f"Text extraction complete! Saved as {output_file}")
