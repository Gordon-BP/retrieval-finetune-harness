# retrieval-finetune-harness
Small Gradio app for fine-tuning document retrieval models

## Changes from the OG Income
* Modified loss function from pairwise subtraction to mean subtraction w. margin
* Using Transformers latest tokenizer for Roberta (like it matters tho)


## OK Here's the plan:
1. Take the multilingual wikipedia embeddings from cohere (like this one https://huggingface.co/datasets/Cohere/miracl-id-corpus-22-12) and put it into a weaviate db
2. Build a basic RAG chat with langchain using cohere embeddings for retrieval and chatGPT for synthesis
3. Take those same multilingual wikipedia embeddings from cohere but yeet the embeddings and build your own IVF_PQ database on weaviate
4. Use https://huggingface.co/doc2query model for the specific language (they have a lot of them) to generate 3 synthetic queries per passage. Should probably just use a random sample of the passages TBH
5. For each query, use a pretrained retriever (msmarco-distilbert-base-tas-b) to pull 25 passages from the corpus.
6. Make triplets out of those passages like (synthetic query, base passage, retrieved passage n) and then run them through a pretrained cross encoder. Use MSE with the difference between encoder loss and cross encoder loss to train.
7. Now you have your fine-tuned model and can run evals!

## Evals!
Since the wikipedia datasets come with queries, you can just take a sample of those for your evals!
1. Sample ~3,000 or so queries
--- Start the timer! ---
2. Encode them:
    - Baseline uses cohere multilingual encoding
    - Experiment uses your finetuned model
3. Do a similarity search in your weaviate database
4. Return top 10 results
5. Eval MAP or MPP at 10, 5, and 1
6. See who is better and cry when it is cohere lol



## To-do
- [x] Find a dataset that you can use as an example
    * use https://huggingface.co/datasets/climate_fever
    * [HARDCORE] semantic wikipedia search https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings
- [x] reject sentence transformers, embrace pytorch
- [x] Modify BiEncoder Pipeline to accept param for top-K
- [ ] Use HF Trainer class
- [ ] Modify Evals to check top-k results, too
- [ ] Modify rerank pipeline to accept biencoder pipeline as param instead of re-defining its methods
- [ ] Chinchilla scaling checker
- [ ] Make a basic Gradio interface
    ✅ Change hyperparameters
    * View evals during training
    * View training data and results data
    * View final test dataset
    * Save hyperparams and results to a csv
- [ ] Better system to save & load models
- [ ] Auto-generate copy/paste custom Langchain retriever class
- [ ] Make it a huggingface space
