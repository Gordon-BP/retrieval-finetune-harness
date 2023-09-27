from src.EmbeddingsGenerator import EmbeddingGenerator

if __name__ == "__main__":
    gen = EmbeddingGenerator("./data/articles_data.csv")
    gen.generate_embeddings_langchain()