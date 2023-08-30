# retrieval-finetune-harness
Small Gradio app for fine-tuning document retrieval models


## To-do
- [x] Find a dataset that you can use as an example
    * use https://huggingface.co/datasets/climate_fever
    * [HARDCORE] semantic wikipedia search https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings
- [ ] reject sentence transformers, embrace pytorch
- [ ] Modify BiEncoder Pipeline to accept param for top-K
- [ ] Modify Evals to check top-k results, too
- [ ] Modify rerank pipeline to accept biencoder pipeline as param instead of re-defining its methods
- [ ] Chinchilla scaling checker
- [ ] Make a basic Gradio interface
    * Change hyperparameters
    * View evals during training
    * View training data and results data
    * View final test dataset
    * Save hyperparams and results to a csv
- [ ] Better system to save & load models
- [ ] Auto-generate copy/paste custom Langchain retriever class
- [ ] Make it a huggingface space
