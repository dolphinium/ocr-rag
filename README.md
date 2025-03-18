# OCR-RAG: Intelligent Document Processing with RAG

## Overview
OCR-RAG is an advanced document processing system that combines Optical Character Recognition (OCR) with Retrieval-Augmented Generation (RAG) to extract, process, and analyze information from images and documents. The system leverages state-of-the-art AI technologies to convert image-based documents into searchable, analyzable text while maintaining context and improving information retrieval accuracy.

## Key Features
- **Advanced OCR Processing**: Utilizes EasyOCR and Google Vision API's text-detection API for robust text extraction from various image formats
- **Intelligent Information Retrieval**: Implements RAG architecture for context-aware document processing
- **Vector-Based Search**: Employs FAISS for efficient similarity search and information retrieval
- **Scalable Architecture**: Modular design supporting various document types and formats
- **HEIF Image Support**: Native support for High Efficiency Image Format files

## Technologies Used

### Core Technologies
- **Python**: Primary programming language
- **EasyOCR**: OCR engine for text extraction
- **Google Vision API**: Google's OCR engine for text extraction (better than EasyOCR on Turkish)
- **OpenAI API**: For advanced language processing and generation
- **FAISS**: Facebook AI Similarity Search for efficient vector search operations

### AI/ML Components
1. **Optical Character Recognition (OCR)**
   - EasyOCR for robust text detection and recognition
   - Google Vision API for text detection
   - Support for multiple languages and scripts
   - Advanced image preprocessing capabilities

2. **Retrieval-Augmented Generation (RAG)**
   - Vector embeddings for semantic search
   - Context-aware information retrieval
   - Dynamic knowledge base integration

3. **Vector Search**
   - FAISS-based similarity search
   - Efficient indexing and retrieval of document vectors
   - Scalable to large document collections

### Supporting Technologies
- **Pillow & Pillow-HEIF**: Image processing and HEIF format support
- **tqdm**: Progress bar functionality for long-running processes
- **python-dotenv**: Environment variable management
- **tiktoken**: Token counting and management for OpenAI models

## Project Structure
```
.
├── src/
│   ├── pipeline.py         # Main processing pipeline
│   ├── RAG/               # RAG implementation
│   └── img-to-text/       # OCR processing modules
├── config/                # Configuration files
├── data/                  # Data storage
└── tests/                # Test suite
```
