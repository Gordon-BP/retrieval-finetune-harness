from src.EmbeddingsGenerator import EmbeddingGenerator

# Chunks your document and then uses OpenAI Ada
# embeddings. This uses the langchain
# implementation, which handles retries and was less
# work for me than adapting the cohere API call method.
# It's fast, but not as fast as cohere.
# But OpenAI is much cheaper!

if __name__ == "__main__":
    gen = EmbeddingGenerator("./data/articles_data.csv")
    gen.generate_embeddings_langchain()