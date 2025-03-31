# ChromaDB Documentation

## Introduction
ChromaDB is an open-source vector database designed to efficiently store and retrieve vector embeddings. It is primarily used for managing embeddings along with metadata, enabling seamless integration with large language models. ChromaDB is particularly useful for applications such as semantic search and knowledge retrieval.

## Features

### Simple and Powerful
- Easily installable via:
  ```sh
  pip install chromadb
  ```
- Quick setup with Python SDK for seamless integration.

### Comprehensive Retrieval
- Supports vector search, full-text search, document storage, metadata filtering, and multi-modal retrieval.
- Can use various storage backends, including DuckDB for local setups and ClickHouse for scalable deployments.

### Multi-Language Support
- Provides SDKs for multiple programming languages, including Python, JavaScript/TypeScript, Ruby, PHP, and Java.

### Seamless Integration
- Compatible with popular embedding models such as HuggingFace, OpenAI, Google, and Jina AI.
- Works with frameworks like Langchain and LlamaIndex.

### Open Source
- Licensed under Apache 2.0, allowing for community-driven development and contributions.

## How ChromaDB Works

### Collection Creation
A collection in ChromaDB is analogous to a table in relational databases. It organizes embeddings and metadata efficiently.
By default, ChromaDB uses `all-MiniLM-L6-v2` for text embedding, but it can be customized to work with other models, such as Jina AI’s `jinaai/jina-embedding-s-en-v1`.

### Adding Data
After creating a collection, text documents with metadata and unique IDs are added. ChromaDB then converts the text into embeddings automatically.

### Querying Data
Users can retrieve data using text-based or embedding-based queries. Metadata filters further refine the search results.

## Getting Started

### Installation
Ensure you have SQLite version 3.35 or higher. If using an older version, upgrade Python to 3.11 or install an older version of ChromaDB.

Install ChromaDB with:
```sh
pip install chromadb
```

### Creating a Collection
In ChromaDB, collections serve as structured containers for storing embeddings.

```python
import chromadb
client = chromadb.PersistentClient()
collection = client.create_collection(name="collection_name")
```

### Adding Documents to a Collection

```python
collection.add(
    documents=["text"],
    metadatas=[{"key": "value"}],
    ids=["unique_id"]
)
```

## Updating and Removing Data

### Updating a Record

```python
collection.update(
    ids=["unique_id"],
    documents=["updated_text"],
    metadatas=[{"key": "updated_value"}]
)
```

### Deleting a Record

```python
collection.delete(ids=["unique_id"])
```

## Collection Management

### Retrieving Documents

```python
collection.get(ids=["unique_id"])
```

### Counting Documents

```python
collection.count()
```

### Modifying a Collection

```python
collection.modify(name="new_collection_name")
```

### Deleting a Collection

```python
client.delete_collection(name="collection_name")
```

### Resetting the Database

```python
client.reset()
```

## Conclusion
ChromaDB is a robust and scalable vector database designed for efficient storage and retrieval of embeddings. With its intuitive API, extensive integration capabilities, and support for various embedding models, it is an ideal choice for AI-driven applications requiring semantic search and knowledge retrieval.

