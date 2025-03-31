import faiss
import numpy as np
import chromadb
import os
import json
import uuid
from embeddings import get_embedding  # Jinnai embedding model
from embeddings import collection  # ChromaDB collection

FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKED_OUTPUT_FOLDER = "./chunked_output"

# ✅ Load embeddings from ChromaDB
def load_embeddings_from_chroma(collection):
    """Fetch embeddings and metadata from ChromaDB."""
    results = collection.get(include=["embeddings", "metadatas", "documents"])  # ✅ Valid fields
    
    if not results or "embeddings" not in results:
        print("❌ No embeddings found in ChromaDB.")
        return None, None, None

    embeddings_np = np.array(results["embeddings"], dtype=np.float32)
    metadata_list = results["metadatas"]
    ids_list = results["documents"]  # Using 'documents' instead of 'ids'
    
    return embeddings_np, metadata_list, ids_list

# ✅ Train FAISS IVF Index (Only needed for large-scale retrieval)
def train_faiss_index(embeddings_np, nlist=100):
    """Trains a FAISS IVF index before adding vectors."""
    d = embeddings_np.shape[1]
    quantizer = faiss.IndexFlatL2(d)  # L2 distance quantizer
    faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    faiss_index.train(embeddings_np)
    print("✅ FAISS IVF Index trained.")
    return faiss_index

# ✅ Build FAISS Index
def build_faiss_index(embeddings_np, use_ivf=False):
    """Builds FAISS index from stored embeddings."""
    if embeddings_np is None or embeddings_np.shape[0] == 0:
        print("❌ No embeddings available to build FAISS index.")
        return None
    
    d = embeddings_np.shape[1]
    
    if use_ivf:
        faiss_index = train_faiss_index(embeddings_np)
    else:
        faiss_index = faiss.IndexFlatL2(d)  # Simple L2 search (no training required)
    
    faiss_index.add(embeddings_np)
    print(f"✅ FAISS Index built with {len(embeddings_np)} vectors.")
    return faiss_index

# ✅ Query FAISS
def query_faiss(faiss_index, collection, query_text, k=5):
    """Search FAISS and retrieve top matches from ChromaDB."""
    if faiss_index is None:
        print("❌ FAISS Index is not initialized.")
        return

    query_embedding, _ = get_embedding(query_text)
    query_embedding_np = np.array([query_embedding], dtype=np.float32)
    distances, indices = faiss_index.search(query_embedding_np, k)
    
    print("\n🔍 Top Matches:")
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        
        result = collection.get(ids=[str(idx)], include=["metadatas"])
        if result and "metadatas" in result and result["metadatas"]:
            metadata = result["metadatas"][0]
            print(f"\n🔹 Match {i+1}:")
            print(f"📄 Chunk ID: {metadata.get('chunk_id', 'Unknown')}")
            print(f"📂 Source: {metadata.get('source', 'Unknown')}")
            print(f"📜 Text: {metadata.get('text', 'No Text')}\n")

# ✅ Store embeddings in FAISS & ChromaDB
def store_chunks_in_faiss():
    """Reads chunked text files, generates embeddings, and stores in FAISS & ChromaDB."""
    json_files = [f for f in os.listdir(CHUNKED_OUTPUT_FOLDER) if f.endswith("_chunked.json")]
    if not json_files:
        print("No chunked JSON files found.")
        return

    embeddings_list = []
    metadata_list = []
    ids_list = []
    faiss_ids_list = []

    for json_file in json_files:
        json_path = os.path.join(CHUNKED_OUTPUT_FOLDER, json_file)
        with open(json_path, "r", encoding="utf-8") as file:
            chunked_data = json.load(file)

        for chunk_key, chunk_text in chunked_data.items():  
            embedding, _ = get_embedding(chunk_text)
            if embedding is None:
                print(f"Skipping {chunk_key} due to embedding failure.")
                continue

            chroma_id = str(uuid.uuid4())  # ChromaDB ID
            faiss_id = len(faiss_ids_list)  # Integer FAISS ID
            
            embeddings_list.append(embedding)
            ids_list.append(chroma_id)
            faiss_ids_list.append(faiss_id)
            metadata_list.append({"chunk_id": chunk_key, "source": json_file, "text": chunk_text, "faiss_id": faiss_id})
    
    embeddings_np = np.array(embeddings_list, dtype=np.float32)
    faiss_index = build_faiss_index(embeddings_np, use_ivf=False)  # Use IVF=True for large datasets
    faiss_index.add_with_ids(embeddings_np, np.array(faiss_ids_list))
    
    collection.add(ids=ids_list, metadatas=metadata_list, documents=[m["text"] for m in metadata_list])
    
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    print("✅ FAISS Index Saved.")
    print("✅ Chunks stored in FAISS & ChromaDB.")

# ✅ Main Execution
if __name__ == "__main__":
    store_chunks_in_faiss()  # Store chunks first
    embeddings_np, metadata_list, ids_list = load_embeddings_from_chroma(collection)
    faiss_index = build_faiss_index(embeddings_np)
    query_faiss(faiss_index, collection, "Partnership is defined as")
