import json
import faiss
import openai
import numpy as np
import tiktoken
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()
# OpenAI API Key
API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = API_KEY

# Define input & output files
jsonl_file = "../../data/processed/book_text_2.jsonl"
faiss_index_file = "./indexes-n-metadata/faiss_index_2.bin"
embedding_dim = 1536  # OpenAI 'text-embedding-ada-002' output size

# Load text data
def load_text_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            page_data = json.loads(line)
            data.append(page_data)
    return data

# Tokenizer to count token lengths
tokenizer = tiktoken.get_encoding("cl100k_base")

# Function to chunk text into smaller segments (max 512 tokens)
def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks, current_chunk = [], []
    current_length = 0

    for word in words:
        token_count = len(tokenizer.encode(word))  # Count tokens
        if current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
        current_chunk.append(word)
        current_length += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Generate embeddings using OpenAI
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response["data"][0]["embedding"], dtype=np.float32)

# Load book text data
book_data = load_text_data(jsonl_file)

# Initialize FAISS index
index = faiss.IndexFlatL2(embedding_dim)

# Store metadata (to retrieve actual text)
metadata = {}

# Process and store embeddings
print("Generating embeddings and storing in FAISS...")
vector_count = 0
for page in tqdm(book_data):
    page_number = page["page"]
    text_chunks = chunk_text(page["text"])
    
    for chunk in text_chunks:
        embedding = get_embedding(chunk)
        index.add(np.array([embedding]))  # Add to FAISS
        metadata[vector_count] = {"page": page_number, "text": chunk}
        vector_count += 1

# Save FAISS index
faiss.write_index(index, faiss_index_file)

# Save metadata separately
with open("./indexes-n-metadata/metadata_2.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("Embedding storage complete! FAISS index and metadata saved.")
