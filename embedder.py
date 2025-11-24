from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.neighbors import NearestNeighbors

embedder_cache = None

def load_embedder(model="all-MiniLM-L6-v2"):
    global embedder_cache
    if embedder_cache is None:
        embedder_cache = SentenceTransformer(model)
    return embedder_cache

def embed_texts(embedder, texts):
    return embedder.encode(texts, convert_to_numpy=True)

def build_index(embeddings):
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(embeddings)
    return nn

def retrieve_top_k(query_emb, nn, k=4):
    distances, indices = nn.kneighbors(query_emb.reshape(1, -1), n_neighbors=k)
    return indices[0]
