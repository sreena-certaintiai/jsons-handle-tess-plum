import chromadb


chroma_client = chromadb.PersistentClient(path="chroma_db_faiss")

collection = chroma_client.get_or_create_collection(
    name="faiss-db"
)