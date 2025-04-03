import json
import faiss
from faiss_chroma import collection  # Your ChromaDB collection

FAISS_INDEX_PATH = "./faiss.index"
FAISS_TO_CHROMA_PATH = "./faiss_to_chroma.json"

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)
ntotal = index.ntotal

# Fetch all IDs from ChromaDB
all_chroma_ids = collection.get()["ids"]

# Generate mapping
faiss_to_chroma_mapping = {str(i): all_chroma_ids[i] for i in range(min(ntotal, len(all_chroma_ids)))}

# Save mapping
with open(FAISS_TO_CHROMA_PATH, "w") as f:
    json.dump(faiss_to_chroma_mapping, f)

print("✅ FAISS-to-Chroma mapping file created successfully!")