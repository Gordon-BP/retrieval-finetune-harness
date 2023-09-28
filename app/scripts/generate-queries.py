from app.src.QueryGenerator import QueryGenerator

if __name__ == "__main__":
    query_gen = QueryGenerator("./data/articles_data.csv")
    query_gen.process_documents()
