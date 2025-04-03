# figure out the token size because there is a change in token calc -> total_token/512 != number of chunks

from transformers import AutoTokenizer
import json
import os
import logging
import time
import pandas as pd
import re

BASE_DIR = r".\\"

# Define paths
JSON_OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_LBG")  # Folder with extracted text
CHUNKED_OUTPUT_FOLDER = os.path.join(BASE_DIR, "chunked_output")  # Folder for chunked text
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")  # Folder to store the CSV report

# Ensure folders exist
os.makedirs(CHUNKED_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(JSON_OUTPUT_FOLDER,exist_ok=True)

# Load JinaAI Tokenizer
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embedding-s-en-v1")

# CSV Report Data
report_data = []

def split_text_into_sentences(text):
    """Splits text into sentences while keeping meaningful segments together."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())  # Split at sentence endings
    return sentences

def tokenize_and_chunk(text, chunk_size=512, overlap=20):
    """Tokenizes text in smaller parts first, then creates exact 512-token chunks with 20-token overlap."""
    sentences = split_text_into_sentences(text)  # Step 1: Pre-split into sentences
    tokens = []
    
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)  
        tokens.extend(sentence_tokens)  # Collect tokens while respecting max length

    # Step 2: Create chunks of exactly 512 tokens with 20-token overlap
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]

        if len(chunk) == chunk_size:  # Ensure exactly 512 tokens
            chunks.append(chunk)

        start += chunk_size - overlap  # Move forward with overlap

    return chunks, len(tokens)  # Return both the chunks and the total token count

def process_json_for_chunking(filename):
    """Loads text from JSON, tokenizes, chunks, saves it, and logs processing time."""
    json_path = os.path.join(JSON_OUTPUT_FOLDER, filename)
    chunked_json_path = os.path.join(CHUNKED_OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_chunked.json")

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        text = data.get("txt_output", "")

    if not text.strip():
        logging.warning(f"Skipping {filename}: No text found.")
        return

    start_time = time.time()  # Start timing

    # Tokenize and chunk the text
    token_chunks, total_tokens = tokenize_and_chunk(text)

    # Decode token chunks back into text
    chunked_output = {}
    base_name = os.path.splitext(filename)[0]  # Extract file name without extension

    for idx, chunk in enumerate(token_chunks, start=1):
        chunk_key = f"{base_name}_chunk_{idx}"  # Example: "document1_chunk_1"
        chunked_output[chunk_key] = tokenizer.decode(chunk, skip_special_tokens=True)

    # Save chunked output as JSON
    with open(chunked_json_path, "w", encoding="utf-8") as chunked_file:
        json.dump(chunked_output, chunked_file, indent=4)

    end_time = time.time()  # End timing
    processing_time = round(end_time - start_time, 2)  # Time taken in seconds

    # Add to report data
    report_data.append([filename, len(token_chunks), processing_time, total_tokens])

    logging.info(f" Chunked data saved to {chunked_json_path}")

def process_all_jsons():
    """Processes all extracted JSON files, tokenizes & chunks them, and generates a CSV report."""
    json_files = [f for f in os.listdir(JSON_OUTPUT_FOLDER) if f.endswith(".json")]

    if not json_files:
        logging.warning(" No JSON files found for chunking.")
        return

    for json_file in json_files:
        process_json_for_chunking(json_file)

    # Save the report as CSV
    if report_data:
        report_df = pd.DataFrame(report_data, columns=["Document Name", "Number of Chunks", "Time Taken (seconds)", "Total Tokens"])
        report_path = os.path.join(REPORT_FOLDER, "chunking_report.csv")
        report_df.to_csv(report_path, index=False)
        logging.info(f" ✅ Chunking report saved to {report_path}")

    logging.info(" ✅ All JSON files have been tokenized and chunked.")

if __name__ == "__main__":
    logging.info(" Starting Tokenization & Chunking...")
    process_all_jsons()
    logging.info(" ✅ Processing complete!")
    print("chunking done")

