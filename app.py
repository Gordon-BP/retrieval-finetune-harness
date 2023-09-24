# @title Main
import numpy as np
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import src.old.ClimateFeverDataLoader as ClimateFeverDataLoader
import src.old.BertTrainingPipeline as BertTrainingPipeline
from src.FinetuningPipeline import FinetuningPipeline

# Data & Evaluation hyperparameters
DATA_SPLIT = 0.7
UNKNOWN_PHRASE = "i don't know the answer to this question"
COSINE_THRESHOLD = 0.5
DOT_THRESHOLD = 0.75
EUC_THRESHOLD = 110
HNSW_THRESHOLD = 0.5
# Init data
evidence_corpus = pd.DataFrame()
bi_encoder_training_set = pd.DataFrame()
cross_encoder_training_set = pd.DataFrame()
eval_set = pd.DataFrame()


def load_climate_data():
    # Logic to load your dataset
    data = ClimateFeverDataLoader.ClimateFeverDataLoaderClass()
    global evidence_corpus
    global bi_encoder_training_set
    global cross_encoder_training_set
    global eval_set
    (
        evidence_corpus,
        bi_encoder_training_set,
        cross_encoder_training_set,
        eval_set,
    ) = data()
    bi_train_limit = np.round(len(bi_encoder_training_set) * DATA_SPLIT)
    ct_train_limit = np.round(cross_encoder_training_set.shape[0] * DATA_SPLIT)
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

def load_mwediawiki_data(path:str)->str:
    loader = MWDumpLoader(
        file_path = path, 
        encoding="utf8",
        #namespaces = [0,2,3] Optional list to load only specific namespaces. Loads all namespaces by default.
        skip_redirects = True, #will skip over pages that just redirect to other pages (or not if False)
        stop_on_error = False #will skip over pages that cause parsing errors (or not if False)
        )
    documents = loader.load()
    print(f"You have {len(documents)} document(s) in your data ")

def train_model(model_name, epochs, batches, margin, learning_rate, warmup_mult):
    global bi_encoder_training_set
    pipeline = FinetuningPipeline(
        run_name="Test run 1",
        model_name="bert-base-uncased",
        epochs=2,
        batch_size=16,
        training_set=bi_encoder_training_set,
    )
    pipeline.finetune_model()
    return "Model trained successfully!"


def evaluation_results():
    # Logic to get the precision, recall, and F1 scores for the four runs
    # Return a dataframe for this data
    df = pd.DataFrame(
        {
            "Run": ["Run1", "Run2", "Run3", "Run4"],
            "Precision": [0.9, 0.91, 0.93, 0.94],  # example values
            "Recall": [0.88, 0.9, 0.89, 0.92],
            "F1": [0.89, 0.905, 0.91, 0.93],
        }
    )
    return df


def download_results():
    # Logic to download the results as CSV
    df = evaluation_results()
    df.to_csv("evaluation_results.csv", index=False)
    return "Results downloaded successfully!"


def fetch_dataset(dataset):
    df_map = {
        "evidence_corpus": evidence_corpus,
        "bi_encoder_training_set": pd.DataFrame(bi_encoder_training_set),
        "cross_encoder_training_set": cross_encoder_training_set,
        "eval_set": eval_set,
    }
    return df_map[dataset].iloc[:15]


with gr.Blocks() as demo:
    gr.Markdown("# Gordy's Text Retriever Fine-Tuning UI")
    with gr.Row():
        gr.Markdown(
            """
                    This is a simple UI for fine-tuning text retrieval systems. See the **Help** tab for more info.
                    """
        )
    with gr.Tab("Load Dataset"):  # Data loading window
        gr.Markdown("## Load Dataset")
        gr.Markdown(
            "Press the 'load data' button first, then use the dropdown to select a set to view"
        )
        data_stats = gr.Markdown()
        load_button = gr.Button(value="Load Data")
        load_button.click(fn=load_climate_data, inputs=[], outputs=[data_stats])
        # gr.DataFrame(fn=fetch_dataset, row_count=10, headers=['index', 'triplet'])
        df = gr.Dropdown(
            label="Dataset selection",
            choices=[
                "evidence_corpus",
                "bi_encoder_training_set",
                "cross_encoder_training_set",
                "eval_set",
            ],
            value="evidence_corpus",
        )
        display_df = gr.DataFrame(
            label="Choose one of the datasets to explore",
            max_rows=15,
            wrap=True,
        )
        df.input(fn=fetch_dataset, inputs=[df], outputs=[display_df])

    with gr.Tab("Model Training"):  # Model training window
        gr.Markdown("## Train a Bi-Encoder Model")
        model_name = gr.Dropdown(
            value="bert-base-uncased",
            choices=["bert-base-uncased", "bert-large-uncased"],
        )  # Add your model names
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Training Hyperparameters")
                batches = gr.Number(
                    value=64,
                    minimum=0,
                    maximum=512,
                    label="BATCH_SIZE",
                    interactive=True,
                )
                epochs = gr.Slider(
                    minimum=1,
                    maximum=16,
                    step=1,
                    value=2,
                    label="BI_ENCODER_EPOCHS",
                    interactive=True,
                )
                learning_rate = gr.Slider(
                    minimum=1e-7,
                    maximum=1e-2,
                    value=5e-3,
                    step=100,
                    label="BI_LEARNING_RATE",
                    interactive=True,
                )
                margin = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=2,
                    step=0.5,
                    label="BI_ENCODER_TRIPLET_MARGIN",
                    interactive=True,
                )
                warmup_mult = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.1,
                    step=0.05,
                    label="BI_ENCODER_WARMUP_MULT",
                    interactive=True,
                )
                steps_mult = gr.Slider(
                    minimum=0,
                    maximum=3,
                    value=1,
                    step=0.5,
                    label="BI_ENCODER_STEPS_MULT",
                    interactive=True,
                )
            with gr.Column():
                gr.Markdown("## Evaluation Hyperparameters")
                cosine = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.5,
                    label="Cosine Similarity Threshold",
                    interactive=True,
                )
                dot = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.75,
                    label="Dot Product Similarity Threshold",
                    interactive=True,
                )
                euc = gr.Slider(
                    minimum=0,
                    maximum=150,
                    step=5,
                    value=100,
                    label="EUC_THRESHOLD",
                    interactive=True,
                )
                hnsw = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.5,
                    label="HNSW_THRESHOLD",
                    interactive=True,
                )

        train_status = gr.Markdown()
        train_button = gr.Button().click(
            fn=train_model,
            inputs=[model_name, epochs, batches, margin, learning_rate, warmup_mult],
            outputs=[train_status],
        )
    # 3. Evaluations window
    with gr.Tab("Model Evaluation"):
        gr.Markdown("## Model Evaluation")
        df = gr.DataFrame(row_count=10)
        gr.DataFrame(label="Evaluation Results")
        download_button = gr.Button(value="Download").click(
            fn=download_results, inputs=[], outputs=[]
        )
    # 4. Model chat!
    with gr.Tab("Model Chat"):

        def echo(message, history):
            return message

        gr.ChatInterface(
            fn=echo, examples=["hello", "hola", "merhaba"], title="Echo Bot"
        )
    # 5. Help window
    with gr.Tab("Help"):
        with open("help.md", "r") as file:
            markdown_content = file.read()
            gr.Markdown(markdown_content)

# Combine the three windows into one interface
demo.launch(debug=True)
