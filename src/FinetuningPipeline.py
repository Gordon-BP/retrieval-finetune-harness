from transformers import (
    AutoTokenizer, 
    AutoModel, 
    DataCollatorForSeq2Seq,
    Trainer, 
    TrainingArguments)
import wandb 
import os
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn import TripletMarginWithDistanceLoss, CosineSimilarity
from .ClimateFeverDataLoader import ClimateFeverDataLoaderClass

class TripletDataset(Dataset):
    def __init__(self, text_data, model_name):
        self.text_data = text_data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.text_data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples, return_tensors='pt', padding="max_length", truncation=True)

    def __getitem__(self, idx):
        anchor, positive, negative = eval(self.text_data[idx])
        anchor_inputs = self.tokenizer(anchor)
        positive_inputs = self.tokenizer(positive)
        negative_inputs = self.tokenizer(negative)
        return [anchor_inputs, positive_inputs, negative_inputs]
    

class FinetuningPipeline:
    """
    Docstring
    """
    def __init__(
            self,
            run_name,
            model_name,
            epochs,
            batch_size,
            training_set,
            wandb_project = "My Project"
        ):
        print("Hello world")

        # download the model and tokenizer
        self.model = AutoModel.from_pretrained(model_name, num_labels=5)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.small_training = TripletDataset(training_set[:1000], model_name)
        self.small_eval = TripletDataset(training_set[1000:300], model_name)
        # set the wandb project where this run will be logged

        os.environ["WANDB_PROJECT"]="my-project"

        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"]="true"

        # turn off watch to log faster
        os.environ["WANDB_WATCH"]="false"

    def compute_metrics(eval_pred, margin=1):
        a,p,n = eval_pred
        loss = TripletMarginWithDistanceLoss(
            distance_function = CosineSimilarity(dim=0),
            margin=margin)
        return loss(a.float(),p.float(),n.float())

    def finetune_model(self):
        # pass "wandb" to the 'report_to' parameter to turn on wandb logging
        training_args = TrainingArguments(
            output_dir='models',
            report_to="wandb",
            logging_steps=5, 
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="steps",
            eval_steps=20,
            max_steps = 100,
            save_steps = 100
        )

        # define our data collator
        my_data_collator = DataCollatorForSeq2Seq(
            tokenizer = self.tokenizer,
            padding = 'longest',
            max_length = 512 #TODO make this based on the model's embeddings length
        )

        # define the trainer and start training
        trainer = Trainer(
            model = self.model,
            tokenizer = self.tokenizer,
            args = training_args,
            train_dataset = self.small_training,
            eval_dataset = self.small_eval,
            data_collator = my_data_collator,
            compute_metrics = self.compute_metrics,
        )
        trainer.train()

        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()