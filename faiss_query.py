import json
import faiss
import numpy as np
from embeddings import get_embedding  # Your embedding function
from faiss_chroma import collection  # Your ChromaDB collection

FAISS_INDEX_PATH = "./faiss.index"
FAISS_TO_CHROMA_PATH = "./faiss_to_chroma.json"  # Mapping file

def load_faiss_to_chroma_mapping():
    """Load the FAISS-to-ChromaDB ID mapping."""
    try:
        with open(FAISS_TO_CHROMA_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ FAISS-to-Chroma mapping file not found!")
        return {}

load_faiss_to_chroma_mapping()

def query_faiss(query_text, k=5):
    """Search FAISS and retrieve metadata from ChromaDB."""
    
    # ✅ Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)

    if index.ntotal == 0:
        print("❌ FAISS index is empty!")
        return
    
    # ✅ Generate embedding for the query text
    query_embedding, _ = get_embedding(query_text)
    query_embedding_np = np.array([query_embedding], dtype=np.float32)

    # ✅ Search FAISS for the top k nearest neighbors
    distances, indices = index.search(query_embedding_np, k)

    # ✅ Load FAISS-to-ChromaDB ID mapping
    faiss_to_chroma_id = load_faiss_to_chroma_mapping()

    print("\n🔍 **Top Matches:**")

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue  # Skip if no match found

        # ✅ Convert FAISS ID to ChromaDB ID
        chroma_id = faiss_to_chroma_id.get(str(idx))  # Convert FAISS index to ChromaDB ID
        
        if not chroma_id:
            print(f"❌ No matching ChromaDB entry for FAISS ID {idx}")
            continue

        # ✅ Retrieve metadata from ChromaDB
        result = collection.get(ids=[chroma_id], include=["metadatas"])

        if result and "metadatas" in result and result["metadatas"]:
            metadata = result["metadatas"][0]
            print(f"\n🔹 **Match {i+1}:**")
            print(f"📄 Chunk ID: {metadata.get('chunk_id', 'Unknown')}")
            print(f"📂 Source: {metadata.get('source', 'Unknown')}")
            print(f"📜 Text: {metadata.get('text', 'No Text')}\n")
        else:
            print(f"❌ No metadata found for ChromaDB ID {chroma_id}")

# Example query
query_faiss("old share = 3 5 Share sacrificed by Ram = 1 4 of 3 3 5 20  Ram\u2019s new share = 3 3 9 5 20 20   Shyam\u2019s old share = 2 5 Share sacrificed by Shyam = 1 3 of 2 2 5 15  Shyam\u2019s new share = 2 2 4 5 15 15   Ghanshyam\u2019s new share = Ram\u2019s sacrifice + Shyam\u2019s Sacrifice = 3 2 17 20 15 60   New profit sharing ratio among Ram, Shyam and Ghanshyam will be 27:16:17. Illustration 5 Das and Sinha are partners in a firm sharing profits in 4:1 ratio. They admitted Pal as a new partner for 1/4 share in the profits, which he acquired wholly from Das. Determine the new profit sharing ratio of the partners. Solution Pal\u2019s share = 1 4 Das\u2019s new share = Old Share \u2013 Share Surrendered = 4 1 5 4  = 11 20 Sinha\u2019s new share = 1 5 The new profit sharing ratio among Das, Sinha and Pal will be 11:4:5. 2.4 Sacrificing Ratio The ratio in which the old partners agree to sacrifice their share of profit in favour of the incoming partner is called sacrificing ratio. The sacrifice by a partner is equal to : Old Share of Profit \u2013 New Share of Profit 2024-25 53 Admission of a Partner As stated earlier, the new partner is required to compensate the old partner\u2019s for their loss of share in the super profits of the firm for which he brings in an additional amount as premium for goodwill. This amount is shared by the existing partners in the ratio in which they forgo their shares in favour of the new partner which is called sacrificing ratio. The ratio is normally clearly given as agreed among the partners which could be the old ratio, equal sacrifice, or a specified ratio. The difficulty arises where the ratio in which the new partner acquires his share from the old partners is not specified. Instead, the new profit sharing ratio is given. In such a situation, the sacrificing ratio is to be worked out by deducting each partner\u2019s new share from his old share. Look at the illustrations 6 to 8 and see how sacrificing ratio is calculated in such a situation. Illustration 6 Rohit and")


