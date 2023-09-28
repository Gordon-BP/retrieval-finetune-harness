from src.EmbeddingsGenerator import EmbeddingGenerator

# Chunks your document and then uses cohere /embed
# endpoint to get embeddings for everything
# It's rather fast! And expensive!
# Cohere is about 4x more $$$ than OpenAI

if __name__ == "__main__":
    gen = EmbeddingGenerator("./data/articles_data.csv")
    gen.generate_embeddings_cohere()