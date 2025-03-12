import os
import json
from google.cloud import vision
from google.oauth2.service_account import Credentials
from tqdm import tqdm

# Define input and output paths
input_folder = "../images_jpg_2"
output_file = "../book_text_2.jsonl"  # JSONL format for chunking in RAG

# Load Google Cloud service account credentials
creds = Credentials.from_service_account_file('../crucial-cabinet-448411-f0-0af14a4cf81e.json')
client = vision.ImageAnnotatorClient(credentials=creds)

# Process images and extract text
book_pages = []

# Sort files to maintain order
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")], key=lambda x: int(x.split('.')[0]))

def detect_text(path):
    with open(path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    
    return texts[0].description if texts else ""

for img_file in tqdm(image_files, desc="Extracting text"):
    img_path = os.path.join(input_folder, img_file)
    text = detect_text(img_path)
    book_pages.append({"page": int(img_file.split(".")[0]), "text": text})

# Save the extracted text in JSONL format
with open(output_file, "w", encoding="utf-8") as f:
    for page in book_pages:
        json.dump(page, f, ensure_ascii=False)
        f.write("\n")

print(f"Text extraction complete! Saved as {output_file}")
