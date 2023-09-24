from src.DataTransformer import DataTransformer
from src.Preprocessor import Preprocessor
from src.JPQModel import JPQModel
from src.GPLTrainingPipeline import GPLTrainingPipeline

# 0. Define common variables
dataset_name = "nfcorpus"
data_dir = f"./datasets/{dataset_name}"
preprocessed_dir = f"./preprocessed/{dataset_name}"


# 1. Preprocess Dataset to JPQ-friendly format
dt = DataTransformer()
dt.transform(
    dataset=dataset_name,
    output_dir=data_dir,
    prefix="",
)

# 2. Preprocessing Script tokenizes the queries and corpus
pp = Preprocessor(
    max_seq_length=512,
    max_query_length=128,
    max_doc_character=10000,
    data_dir=data_dir,
    tokenizer="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    out_data_dir=preprocessed_dir,
    threads=32,
)
pp.preprocess()

# 3. INIT script trains the IVFPQ corpus faiss index
model = JPQModel(
    preprocess_dir=preprocessed_dir,
    model_dir="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    backbone="distilbert",
    output_dir=f"./init/{dataset_name}",
    subvector_num=96,
    max_doc_length=350,
    eval_batch_size=128,
    doc_embed_size=768,
)

# 4. TRAIN script trains TAS-B using Generative Pseudo Labeling (GPL)
jpq_pipeline = GPLTrainingPipeline(
    log_dir="./logs",
    model_save_dir=f"./final_models/{dataset_name}/gpl",
    init_index_path=f"./init/{dataset_name}/OPQ96,IVF1,PQ96x8.index",
    init_model_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    preprocess_dir=preprocessed_dir,
    data_path=data_dir,
)
jpq_pipeline.train_gpl()

# 5. Convert TAS-B trained model into JPQTower (Reqd. for inference)
# python -m income.jpq.models.jpqtower_converter \
#        --query_encoder_model "./final_models/${dataset}/genq/epoch-1" \
#        --doc_encoder_model "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" \
#        --query_faiss_index "./final_models/${dataset}/genq/epoch-1/OPQ96,IVF1,PQ96x8.index" \
#        --doc_faiss_index "./init/${dataset}/OPQ96,IVF1,PQ96x8.index" \
#        --model_output_dir "./jpqtower/${dataset}/"
