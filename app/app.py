# @title Main
import numpy as np
import gradio as gr
import pandas as pd
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.LocalHTMLDataloader import LocalHTMLDataloader

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


def load_data() -> None:
    # Logic to load your dataset
    en_chunks = LocalHTMLDataloader(
        data_dir="./data/en", chunk_size=350, chunk_overlap=55
    )
    bm_chunks = LocalHTMLDataloader(
        data_dir="./data/bm", chunk_size=350, chunk_overlap=55
    )


def get_dataset_info(dataset: str) -> str:
    with open("./src/datasets.json") as file:
        datasets = json.load(file)
        print(datasets[dataset]["languages"])
        language.update(choices=datasets[dataset]["languages"])
        return datasets[dataset]["desc"], datasets[dataset]["languages"]


def fetch_responses(
    query: str,
    language: str,
    chatGPT_response: List[Tuple[str]],
    cohere_response: List[Tuple[str]],
    jank_response: List[Tuple[str]],
) -> List[str]:
    chatGPT_response.append((query, query))
    cohere_response.append((query, query))
    jank_response.append((query, query))
    return "", chatGPT_response, cohere_response, jank_response


with gr.Blocks() as demo:
    gr.Markdown("# Gordy's Multilingual Retrievers")
    with gr.Row():
        gr.Markdown(
            """
            Finetuned retrievers for multilingual RAG: Fast, High-performance, and f**kin tiny!
            """
        )
    with gr.Tab("Demo"):  # Demo Tab
        with gr.Row():
            with open("./src/datasets.json") as file:
                datasets = json.load(file)
                dataset = gr.Dropdown(
                    choices=list(datasets.keys()),
                    value=list(datasets.keys())[2],
                    label="Dataset",
                    interactive=True,
                    scale=1,
                )
                dataset_desc = gr.Textbox(
                    label="About the dataset",
                    value=datasets[dataset.value]["desc"],
                    interactive=False,
                    show_label=True,
                    scale=3,
                )

        with gr.Row():
            chatGPT_response = gr.Chatbot(
                [
                    (
                        "Hi OpenAI",
                        "This chat uses default OpenAI embeddings to retrieve documents",
                    )
                ],
                label="OpenAI Embeddings",
                show_label=True,
                interactive=False,
            )
            cohere_response = gr.Chatbot(
                [
                    (
                        "Hi Cohere",
                        "This chat uses Cohere English or Multilingual embeddings to retrieve documents",
                    )
                ],
                label="Cohere Embeddings",
                show_label=True,
                interactive=False,
            )
            jank_response = gr.Chatbot(
                [
                    (
                        "Hi Jank",
                        "This chat uses fine-tuned embeddings to retrieve documents",
                    )
                ],
                label="Jank Embeddings",
                show_label=True,
                interactive=False,
            )
        with gr.Row():
            language = gr.Dropdown(
                choices=datasets[dataset.value]["languages"],
                value="en",
                label="Language",
                interactive=True,
            )
            composer = gr.Textbox(
                placeholder="ask your question here", interactive=True, scale=4
            )
            submit = gr.Button(value="Submit", interactive=True, scale=1)
            composer.submit(
                fetch_responses,
                [composer, language, chatGPT_response, cohere_response, jank_response],
                [composer, chatGPT_response, cohere_response, jank_response],
            )
            submit.click(
                fetch_responses,
                [composer, language, chatGPT_response, cohere_response, jank_response],
                [composer, chatGPT_response, cohere_response, jank_response],
            )
            dataset.select(get_dataset_info, [dataset], [dataset_desc, language])

    with gr.Tab("About"):  # Model training window
        gr.Markdown("This is where I nerd the fuck out about the tech stack")
    # Initialize the dataset and models on load
    demo.load(load_data)
# Combine the three windows into one interface

demo.launch(debug=True)
