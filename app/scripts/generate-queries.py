from app.src.QueryGenerator import QueryGenerator

# This script will perform document expansion on the dataset
# It uses a T5 model, fine-tuned on the doc's language to generate
# 5 queries per chunk. 
# We use these synthetic queries for domain adaption fine-tuning
# This takes a lot of power!
# Run this in an env with a GPU or you will die of old age before it is done

if __name__ == "__main__":
    query_gen = QueryGenerator("./data/articles_data.csv")
    query_gen.process_documents()
