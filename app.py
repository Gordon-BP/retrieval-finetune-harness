from src.ClimateFeverDataLoader import ClimateFeverDataLoader
import numpy as np
import gradio as gr
import pandas as pd
# Data & Evaluation hyperparameters
DATA_SPLIT = 0.7
UNKNOWN_PHRASE = "i don't know the answer to this question"
COSINE_THRESHOLD = 0.5
DOT_THRESHOLD = 0.75
EUC_THRESHOLD = 110
HNSW_THRESHOLD = 0.5
# Bi-Encoder Hyperparameters
BI_ENCODER_MODEL = 'bert-base-uncased'
BI_ENCODER_BATCH = 64
BI_ENCODER_EPOCHS = 2
BI_ENCODER_TRIPLET_MARGIN = 2
BI_ENCODER_WARMUP_MULT = 0.1 #Warm up on 10% of the data
BI_ENCODER_STEPS_MULT = 2 #How many times to iterate over the data per epoch
evidence_corpus = pd.DataFrame()
bi_encoder_training_set = pd.DataFrame()
cross_encoder_training_set = pd.DataFrame()
eval_set = pd.DataFrame()

def load_dataset():
    # Logic to load your dataset
    data = ClimateFeverDataLoader()
    evidence_corpus, bi_encoder_training_set, cross_encoder_training_set, eval_set = data()
    bi_train_limit = np.round(len(bi_encoder_training_set)*DATA_SPLIT)
    ct_train_limit = np.round(cross_encoder_training_set.shape[0]*DATA_SPLIT)
    data_summary = f"""
    Here are the data stats:
    **Total Facts:** {evidence_corpus.shape[0]}
    **Total Triplets:** {len(bi_encoder_training_set)}
    Total Labeled Claim/Evidence Pairs: {cross_encoder_training_set.shape[0]}

    Only training on the first {DATA_SPLIT} entries, meaning...
    - Bi training set size: {bi_train_limit}
    - Cross training set size: {ct_train_limit}
    - Eval set size: {eval_set.shape[0]}

    """
    return data_summary
def train_model():
    # Logic to train your model using selected hyperparameters
    # Return a message for completion
    return "Model trained successfully!"
def evaluation_results():
    # Logic to get the precision, recall, and F1 scores for the four runs
    # Return a dataframe for this data
    df = pd.DataFrame({
        "Run": ["Run1", "Run2", "Run3", "Run4"],
        "Precision": [0.9, 0.91, 0.93, 0.94],  # example values
        "Recall": [0.88, 0.9, 0.89, 0.92],
        "F1": [0.89, 0.905, 0.91, 0.93]
    })
    return df

def download_results():
    # Logic to download the results as CSV
    df = evaluation_results()
    df.to_csv("evaluation_results.csv", index=False)
    return "Results downloaded successfully!"

with gr.Blocks() as page:
    with gr.Tab("Load Dataset"): #Data loading window
        gr.Markdown("## Load Dataset")
        data_stats = gr.Markdown()
        load_button = gr.Button(value="Load Data")
        load_button.click(fn=load_dataset, inputs=[], outputs=[data_stats])
        gr.DataFrame(bi_encoder_training_set, row_count=10, headers=['index', 'triplet'])
    with gr.Tab("Model Training"): #Model training window
        gr.Markdown("## Train a Bi-Encoder Model")
        gr.Dropdown()
        model_dropdown = gr.inputs.Dropdown(choices=["bert-base-uncased", "bert-large-uncased"])  # Add your model names
        COSINE_THRESHOLD = gr.Slider(minimum=0, maximum=1, default=0.5, label="Cosine Similarity Threshold")
        DOT_THRESHOLD = gr.Slider(minimum=0, maximum=1, default=0.75, label="Dot Product Similarity Threshold")
        EUC_THRESHOLD = gr.Slider(minimum=0, maximum=150, default=110, label="EUC_THRESHOLD")
        HNSW_THRESHOLD = gr.Slider(minimum=0, maximum=1, default=0.5, label="HNSW_THRESHOLD")
        BI_ENCODER_BATCH = gr.Number(default=64, label="Batch Size")
        BI_ENCODER_EPOCHS = gr.Slider(minimum=1, maximum=16, default=2, label="BI_ENCODER_EPOCHS")
        BI_ENCODER_TRIPLET_MARGIN = gr.Slider(minimum=0, maximum=10, default=2, label="BI_ENCODER_TRIPLET_MARGIN")
        BI_ENCODER_WARMUP_MULT = gr.Slider(minimum=0, maximum=1, default=0.1, label="BI_ENCODER_WARMUP_MULT")
        BI_ENCODER_STEPS_MULT = gr.Slider(minimum=0, maximum=3, default=2, label="BI_ENCODER_STEPS_MULT")
        train_status = gr.Markdown()
        train_button = gr.Button(fn=train_model,inputs=[], outputs=[train_status])

# 3. Evaluations window
    with gr.Tab("Model Evaluation"):
        gr.Markdown("## Model Evaluation")
        df = gr.DataFrame(row_count=10)
        gr.DataFrame(fn=evaluation_results, inputs=[], outputs=[df])
        download_button = gr.Button(value="Download").click(fn=download_results, inputs=[], outputs=[])

# Combine the three windows into one interface
page.launch()