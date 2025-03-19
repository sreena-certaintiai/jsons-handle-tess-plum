import json
import os
import spacy
import time
import requests
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure stdout supports UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load NLP model for stopwords removal
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5_000_000  

# Define correct file paths
BASE_DIR = r"C:\Users\Sriharan\Desktop\storage-tesseract"
INPUT_FOLDER = os.path.join(BASE_DIR, "output_LBG")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_chunk")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")

# Ensure output & log folders exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Function to write logs
def write_log(log_file, message):
    """Writes a log entry to a file with a timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_message = f"[{timestamp}] {message}\n"
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(log_message)
    print(log_message.strip())  # Also print to console

# to-do dont remove stopwords for now for testing
# Function to remove stopwords
def remove_stopwords(text):
    """Removes stopwords, punctuation, and spaces using Spacy."""
    start_time = time.time()
    doc = nlp(text)
    processed_text = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space])
    return processed_text, time.time() - start_time  # Return text & processing time

# Function to chunk text into 512 words while maintaining sentence boundaries
def chunk_by_words(text, chunk_size=512, overlap=20):
    """Splits text into chunks of exactly 512 words while maintaining sentence boundaries."""
    start_time = time.time()
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        sentence_length = len(words)

        if current_length + sentence_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk[:chunk_size]))  
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = len(current_chunk)

        current_chunk.extend(words)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk[:chunk_size]))  

    return chunks, time.time() - start_time  # Return chunks & processing time

# Embedding function with retry logic
def get_embedding(text, retries=0):
    start_time = time.time()
    try:
        if not isinstance(text, str) or not text.strip():
            return None, 0  # Skip empty text

        embedding_endpoint = os.getenv("EMBEDDING_API_ENDPOINT")
        if not embedding_endpoint:
            return None, 0  # Skip if endpoint is missing

        response = requests.post(url=embedding_endpoint, json={"text": text})
        if response.status_code != 200:
            return None, 0

        result = response.json()
        if "embedding" not in result or not result.get("embedding"):
            return None, 0

        return result["embedding"][0], time.time() - start_time  # Return embedding & processing time

    except Exception:
        if retries < 3:
            return get_embedding(text, retries=retries + 1)  # Retry
        return None, 0  # Return None if retries exceeded

# Process each JSON file in the folder
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".json"):  # Process only JSON files
        input_file = os.path.join(INPUT_FOLDER, filename)
        output_file = os.path.join(OUTPUT_FOLDER, f"{filename[:-5]}-embeddings.json")  # Remove ".json" from name
        log_file = os.path.join(LOG_FOLDER, f"{filename[:-5]}-log.txt")  # Separate log for each file

        write_log(log_file, f"=== Processing {filename} ===")

        # Load JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)

        # Extract text
        raw_text = extracted_data.get("txt_output", "").strip()
        if not raw_text:
            write_log(log_file, f"Error: No text found in {filename}. Skipping.")
            continue

        # Remove stopwords
        processed_text, stopword_time = remove_stopwords(raw_text)
        write_log(log_file, f"Stopword removal completed in {stopword_time:.2f} seconds.")

        # Chunk text
        chunks, chunking_time = chunk_by_words(processed_text)
        write_log(log_file, f"Chunking completed in {chunking_time:.2f} seconds. {len(chunks)} chunks created.")

        # Generate embeddings
        total_embedding_time = 0
        chunk_data = []
        for i, chunk in enumerate(chunks):
            embedding, embedding_time = get_embedding(chunk)
            total_embedding_time += embedding_time
            if embedding is None:
                write_log(log_file, f"Skipping chunk {i+1} due to embedding failure.")
                continue

            chunk_data.append({"chunk": chunk, "embedding": embedding})
            write_log(log_file, f"Chunk {i+1}/{len(chunks)} embedded successfully in {embedding_time:.4f} seconds.")

        write_log(log_file, f"Total embedding time: {total_embedding_time:.2f} seconds.")

        # Save embeddings
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunk_data}, f, indent=4)

        write_log(log_file, f"{len(chunk_data)} chunks with embeddings saved to {output_file}")
        write_log(log_file, f"=== Completed processing {filename} ===\n")