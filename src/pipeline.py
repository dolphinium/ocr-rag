import os
import json
import numpy as np
import faiss
import openai
import tiktoken
from tqdm import tqdm
from PIL import Image
from pillow_heif import open_heif
import easyocr
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCRPipeline:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the OCR Pipeline with configuration."""
        self.config = config
        self.setup_directories()
        self.setup_openai()
        
        # Initialize OCR reader
        self.reader = easyocr.Reader(['tr'])  # Adjust language as needed
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # FAISS dimension for OpenAI embeddings
        self.embedding_dim = 1536

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.config['jpg_output_dir'],
            os.path.dirname(self.config['jsonl_output']),
            os.path.dirname(self.config['faiss_index_path'])
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")

    def setup_openai(self):
        """Set up OpenAI API key."""
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables")

    def convert_heic_to_jpg(self):
        """Convert HEIC images to JPG format."""
        logger.info("Starting HEIC to JPG conversion...")
        
        heic_files = sorted([
            f for f in os.listdir(self.config['heic_input_dir'])
            if f.lower().endswith('.heic')
        ])
        
        for idx, file in enumerate(tqdm(heic_files, desc="Converting HEIC to JPG")):
            input_path = os.path.join(self.config['heic_input_dir'], file)
            output_path = os.path.join(self.config['jpg_output_dir'], f"{idx + 1}.jpg")
            
            try:
                heif_file = open_heif(input_path)
                image = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                    heif_file.mode
                )
                image.save(output_path, format="JPEG")
            except Exception as e:
                logger.error(f"Error converting {file}: {str(e)}")

        logger.info("HEIC to JPG conversion completed")

    def extract_text(self) -> List[Dict[str, Any]]:
        """Extract text from JPG images using EasyOCR."""
        logger.info("Starting text extraction...")
        
        image_files = sorted(
            [f for f in os.listdir(self.config['jpg_output_dir']) if f.endswith('.jpg')],
            key=lambda x: int(x.split('.')[0])
        )
        
        book_pages = []
        for img_file in tqdm(image_files, desc="Extracting text"):
            img_path = os.path.join(self.config['jpg_output_dir'], img_file)
            try:
                result = self.reader.readtext(img_path)
                text = "\n".join([detection[1] for detection in result])
                book_pages.append({
                    "page": int(img_file.split(".")[0]),
                    "text": text
                })
            except Exception as e:
                logger.error(f"Error processing {img_file}: {str(e)}")
        
        return book_pages

    def save_jsonl(self, book_pages: List[Dict[str, Any]]):
        """Save extracted text to JSONL format."""
        logger.info("Saving extracted text to JSONL...")
        
        with open(self.config['jsonl_output'], "w", encoding="utf-8") as f:
            for page in book_pages:
                json.dump(page, f, ensure_ascii=False)
                f.write("\n")

    def chunk_text(self, text: str, max_tokens: int = 512) -> List[str]:
        """Chunk text into smaller segments."""
        words = text.split()
        chunks, current_chunk = [], []
        current_length = 0

        for word in words:
            token_count = len(self.tokenizer.encode(word))
            if current_length + token_count > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0
            current_chunk.append(word)
            current_length += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(response["data"][0]["embedding"], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def process_embeddings(self, book_pages: List[Dict[str, Any]]):
        """Process text chunks and store embeddings in FAISS."""
        logger.info("Generating embeddings and storing in FAISS...")
        
        # Initialize FAISS index
        index = faiss.IndexFlatL2(self.embedding_dim)
        metadata = {}
        
        vector_count = 0
        for page in tqdm(book_pages, desc="Processing embeddings"):
            text_chunks = self.chunk_text(page["text"])
            
            for chunk in text_chunks:
                try:
                    embedding = self.get_embedding(chunk)
                    index.add(np.array([embedding]))
                    metadata[vector_count] = {
                        "page": page["page"],
                        "text": chunk
                    }
                    vector_count += 1
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
        
        # Save FAISS index and metadata
        faiss.write_index(index, self.config['faiss_index_path'])
        with open(self.config['metadata_path'], "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def run_pipeline(self):
        """Execute the complete pipeline."""
        try:
            # Step 1: Convert HEIC to JPG
            self.convert_heic_to_jpg()
            
            # Step 2: Extract text from images
            book_pages = self.extract_text()
            
            # Step 3: Save extracted text to JSONL
            self.save_jsonl(book_pages)
            
            # Step 4: Process and store embeddings
            self.process_embeddings(book_pages)
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    # Configuration
    config = {
        'heic_input_dir': '../../data/raw/images/images_heic',
        'jpg_output_dir': '../../data/raw/images/images_jpg',
        'jsonl_output': '../../data/processed/book_text.jsonl',
        'faiss_index_path': './RAG/indexes-n-metadata/faiss_index.bin',
        'metadata_path': './RAG/indexes-n-metadata/metadata.json'
    }
    
    # Initialize and run pipeline
    pipeline = OCRPipeline(config)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 