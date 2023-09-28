from src.EmbeddingsGenerator import EmbeddingGenerator

# I haven't actually tested this one...

if __name__ == "__main__":
    gen = EmbeddingGenerator("./data/articles_data.csv")
    gen.generate_embeddings_hf()