import chromadb

# Initialize ChromaDB (persistent mode)
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Local storage

# Create or get collection
collection = chroma_client.get_or_create_collection(name="text_embeddings")


import json
import os

# Path to JSON embeddings
OUTPUT_FOLDER = r"C:\Users\Sriharan\Desktop\storage-tesseract\output_chunk"

# Load embeddings into ChromaDB
for filename in os.listdir(OUTPUT_FOLDER):
    if filename.endswith("-embeddings.json"):
        with open(os.path.join(OUTPUT_FOLDER, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            for i, chunk_data in enumerate(data["chunks"]):
                collection.add(
                    ids=[f"{filename}-{i}"],  # Unique ID
                    embeddings=[chunk_data["embedding"]],  # Store only embeddings
                    metadatas=[{"json_filename": filename, "chunk_index": i}]  # Link back to JSON
                )

print("Embeddings uploaded to ChromaDB!")


# to-do change the embedding function

import requests
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()

def get_query_embedding(query_text):
    """Generates an embedding for a user query."""
    embedding_endpoint = os.getenv("EMBEDDING_API_ENDPOINT")
    if not embedding_endpoint:
        return None

    response = requests.post(url=embedding_endpoint, json={"text": query_text})
    if response.status_code != 200:
        return None

    result = response.json()
    return result.get("embedding", [])[0]  # Extract embedding


query="aim to find Prime numbers"
# query_vector = get_query_embedding(query)
# print(query_vector)

def search_chroma(query_text, top_k=5):
    """Finds the closest stored embeddings in ChromaDB."""
    query_vector = get_query_embedding(query_text)
    if not query_vector:
        print("Error: Failed to generate query embedding.")
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k  # Number of matches
    )

    return results["metadatas"][0]  # Return metadata of closest matches

# result=search_chroma(query, top_k=3)
# print(result)

def fetch_chunks_from_json(search_results):
    """Fetches text chunks from stored JSON files based on search results."""
    retrieved_chunks = []

    for result in search_results:
        filename = result["json_filename"]
        index = result["chunk_index"]

        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                chunk_data = data["chunks"][index]["chunk"]  # Extract chunk
                retrieved_chunks.append({"chunk": chunk_data, "source": filename})

    return retrieved_chunks

# results2 = fetch_chunks_from_json(result)
# print(results2)

def query_pipeline(query_text, top_k=5):
    """Complete pipeline: get embedding → search ChromaDB → fetch text from JSON."""
    search_results = search_chroma(query_text, top_k)
    matching_chunks = fetch_chunks_from_json(search_results)

    return matching_chunks

# Example Usage:
query = "Sonya Nicholas said "
results = query_pipeline(query, top_k=1)

for res in results:
    print(f"Source: {res['source']}\nChunk: {res['chunk']}\n---")
