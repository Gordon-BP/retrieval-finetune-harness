from transformers import Trainer
from torch.nn import TripletMarginLoss
import os


class CustomTrainerClass(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        margin = 2 if not os.environ["TRIPLET_MARGIN"] else os.environ["TRIPLET_MARGIN"]
        loss = TripletMarginLoss(margin=margin)
        return loss(inputs[0], inputs[1], inputs[2])
