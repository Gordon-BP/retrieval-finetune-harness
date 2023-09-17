import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from torch.nn import TripletMarginLoss
from tqdm import tqdm


#
# Wow look at me I'm so smart I can build my own model training pipeline
# using only pytorch! It only took me a week and it is both slower and
# less feature-rich than the myriad pre-built and widely adopted solutions!
# I am soooo proud of myself
#
class CustomDataset(Dataset):
    def __init__(self, text_data, model_name):
        self.text_data = text_data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        anchor, positive, negative = eval(self.text_data[idx])
        anchor_inputs = self.tokenizer(
            anchor, return_tensors="pt", truncation=True, padding=False
        )
        positive_inputs = self.tokenizer(
            positive, return_tensors="pt", truncation=True, padding=False
        )
        negative_inputs = self.tokenizer(
            negative, return_tensors="pt", truncation=True, padding=False
        )
        return anchor_inputs, positive_inputs, negative_inputs


class BiEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(BiEncoderModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask=attention_mask).pooler_output


class BertTrainingPipeline:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = BiEncoderModel(model_name)

    def collate_fn(self, batch):
        anchor_batch, positive_batch, negative_batch = zip(*batch)
        anchor_input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"].squeeze() for item in anchor_batch], batch_first=True
        )
        anchor_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"].squeeze() for item in anchor_batch],
            batch_first=True,
        )
        positive_input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"].squeeze() for item in positive_batch], batch_first=True
        )
        positive_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"].squeeze() for item in positive_batch],
            batch_first=True,
        )
        negative_input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"].squeeze() for item in negative_batch], batch_first=True
        )
        negative_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"].squeeze() for item in negative_batch],
            batch_first=True,
        )
        return {
            "anchor": {
                "input_ids": anchor_input_ids,
                "attention_mask": anchor_attention_mask,
            },
            "positive": {
                "input_ids": positive_input_ids,
                "attention_mask": positive_attention_mask,
            },
            "negative": {
                "input_ids": negative_input_ids,
                "attention_mask": negative_attention_mask,
            },
        }

    def lr_lambda(self, step, total_steps, warmup_mult):
        warmup_steps = int(total_steps * warmup_mult)
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            return 1

    def train(
        self,
        BI_ENCODER_EPOCHS,
        BI_ENCODER_TRIPLET_MARGIN,
        BI_ENCODER_LEARNING_RATE,
        BI_ENCODER_WARMUP_MULT,
        BI_ENCODER_BATCH,
        bi_encoder_training_set,
        PROGRESS_BAR,
    ):
        # Load data
        train_dataset = CustomDataset(
            bi_encoder_training_set[1:].tolist(), self.model_name
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=BI_ENCODER_BATCH,
        )

        # Initialize model, loss, optimizer
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = TripletMarginLoss(margin=BI_ENCODER_TRIPLET_MARGIN)
        optimizer = optim.AdamW(model.parameters(), lr=BI_ENCODER_LEARNING_RATE)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: self.lr_lambda(x, len(train_dataloader), BI_ENCODER_WARMUP_MULT),
        )

        # Training loop
        model.train()
        num_epochs = BI_ENCODER_EPOCHS
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(train_dataloader):
                print(batch)
                optimizer.zero_grad()
                anchor_input_ids = batch["anchor"]["input_ids"]
                anchor_attention_mask = batch["anchor"]["attention_mask"]
                positive_input_ids = batch["positive"]["input_ids"].squeeze()
                positive_attention_mask = batch["positive"]["attention_mask"]
                negative_input_ids = batch["negative"]["input_ids"].squeeze()
                negative_attention_mask = batch["negative"]["attention_mask"]
                anchor = model(anchor_input_ids, anchor_attention_mask)
                positive = model(positive_input_ids, positive_attention_mask)
                negative = model(negative_input_ids, negative_attention_mask)
                loss = criterion(anchor, positive, negative)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}"
            )

        # Save the model
        torch.save(model.state_dict(), "./bi_encoder")

        print("Model is done training!")
