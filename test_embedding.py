import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env
load_dotenv()

local_model_path = "./bge-small-local"
embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

if not Path(local_model_path).exists():
    print(f"Downloading model weights ({embedding_model}) to local directory: {local_model_path}")
    snapshot_download(repo_id=embedding_model, local_dir=local_model_path)

# BGE models formatting variables for optimal performance
encode_kwargs = {'normalize_embeddings': True}  # Required for BGE v1.5 cosine similarity
query_kwargs = {'prompt': 'Represent this sentence for searching relevant passages: '}

embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path,
    encode_kwargs=encode_kwargs,
    query_encode_kwargs=query_kwargs
)

# Test the model
vector = embeddings.embed_query("Testing my local BGE setup.")
print("First 5 dimensions:", vector[:5])
print("Total vector dimension:", len(vector))
