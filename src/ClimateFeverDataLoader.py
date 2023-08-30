from typing import List, Tuple
import os
import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel

class Triplet(BaseModel):
    """
    Class defining a triplet datapoint.
    It's not labeled or enforced, but the triplet order is:
    anchor, positive example, negative example
    """
    items: List[str]

class CrossEncoderDatum(BaseModel):
    texts: List[str]
    label: int


class ClimateFeverDataLoader():
    """
    Class to load the Climate Fever dataset and then parse it into four datasets:
        * Evidence Corpus: a DataFrame of all the different pieces of evidence (facts) used
        * Bi Encoder Training: A list of triplets build from the dataset
        * Cross Encoder Training: A list of claims, evidence, and labels from the dataset
        * Eval Dataset: A dataframe with claims, labels, and expected evidence
    """

    def makeEvidenceCorpus(self, dataset:pd.DataFrame)-> pd.DataFrame:
        """
        Makes a DataFrame with all the unique pieces of evidence

        Args:
        - dataset (pd.DataFrame): the raw dataset, loaded as a DataFrame from HuggingFace

        Returns:
        pd.DataFrame the evidence corpus as a small DataFrame. Looks like this:
        idx | evidence |
        """
        def extractEvidence(row):
            return [_['evidence'].replace("\"","") for _ in row['evidences']]
        evidence_corpus = pd.DataFrame()
        evidence_corpus['evidence'] = [item for sublist in dataset.apply(extractEvidence, axis=1).tolist() for item in sublist]
        evidence_corpus.to_csv("data/evidence_corpus.csv")
        return evidence_corpus
        
    def makeBiTrain(self, dataset:pd.DataFrame, evidence_corpus:pd.DataFrame)->List[Triplet]:
        """
        Makes the Bi Encoder training data

        Args:
        - dataset (pd.DataFrame): the raw dataset, loaded as a DataFrame from HuggingFace
        - evidence_corpus (pd.DataFrame): a DataFrame of all the unique pieces of evidence

        Returns:
        List[Triplets]: Training data for the bi-encoder. Looks like this:
        [[claim, relevant evidence, irrelevant evidence],...]
        """
        def make_triplets(row):
            if(row['claim_label'] == 2):
                return None
            else:
                triplets = []
                relevant_evidence = []
                irrelevant_evidence = []
                for e in row['evidences']:
                    if(e['evidence_label'] == row['claim_label']):
                        relevant_evidence.append(e['evidence'])
                    else:
                        irrelevant_evidence.append(e['evidence'])
                if(len(irrelevant_evidence)!= 0):
                    for re in relevant_evidence:
                        for ie in irrelevant_evidence:
                            triplets.append(Triplet(items=[row['claim'], re, ie]))
                else:
                    ie = evidence_corpus.loc[~evidence_corpus['evidence'].isin(relevant_evidence)]
                    for re in relevant_evidence:
                        triplets.append(Triplet(items=[row['claim'], re, ie.sample().iloc[0][0]]))
                return triplets
        
        trips = [trip for trip in dataset.apply(make_triplets, axis=1) if trip is not None]
        trips_df = pd.DataFrame(trips)
        trips_df.to_csv("data/bi_encoder_training")
        return trips_df
    def makeCrossTrain(self, dataset: pd.DataFrame)-> pd.DataFrame:
        """
        Makes the cross encoder training data

        Args:
        - dataset (pd.DataFrame): the raw dataset, loaded as a DataFrame from HuggingFace

        Returns:
        pd.DataFrame: A Dataframe of training points for the cross encoder. Looks like this:
        idx | texts [claim, evidence] | label (0, 1, 2)|
        """
        def cross_train_samples(row):
            samples = []
            for e in row['evidences']:
                samples.append(CrossEncoderDatum(**{"texts":[row['claim'], e['evidence']], "label":e['evidence_label']}))
            return samples

        cross_samples = [item for sublist in dataset.apply(cross_train_samples, axis=1) for item in sublist]
        cross_df = pd.DataFrame(cross_samples)
        cross_df.to_csv("data/cross_encoder_training.csv")
        return cross_df
    def makeEvalSet(self, dataset:pd.DataFrame)->pd.DataFrame:
        """
        Makes the initial eval dataset

        Args:
        - dataset (pd.DataFrame): the raw dataset, loaded as a DataFrame from HuggingFace

        Returns:
        The expected parts of the eval set, looks like:
        idx | claim | expected determination | expected relevant evidence |
        """
        def make_eval(row):
            supporting_evidence = []
            for e in row['evidences']:
                if(e['evidence_label'] == row['claim_label']):
                    supporting_evidence.append(e['evidence'])
            return {
                "claim":row['claim'],
                "expected determination":row['claim_label'],
                "expected relevant evidence":supporting_evidence
            }
        eval_df = pd.DataFrame.from_records(dataset.apply(make_eval, axis=1))
        eval_df.to_csv("data/evidence_corpus.csv")
        return eval_df
    
    def __call__(self)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_path = "./data"
        required_files = [
            "evidence_corpus.csv",
            "bi_encoder_training.csv",
            "cross_encoder_training.csv",
            "eval_set.csv"
        ]
        missing_files = []
        for filename in required_files:
            file_path = os.path.join(data_path, filename)
            if not os.path.isfile(file_path):
                missing_files.append(filename)

        if missing_files: #Just re-load everything if missing
            dataset = pd.DataFrame(load_dataset("climate_fever", split='test'))
            evidence_corpus = self.makeEvidenceCorpus(dataset)
            bi_encoder_training_set = self.makeBiTrain(dataset, evidence_corpus)
            cross_encoder_training_set = self.makeCrossTrain(dataset)
            eval_set = self.makeEvalSet(dataset)
            dataset = None #clean up disk space
        else:
            evidence_corpus = pd.read_csv(os.path.join(data_path, required_files[0]))
            bi_encoder_training_set = pd.read_csv(os.path.join(data_path, required_files[1]))
            cross_encoder_training_set = pd.read_csv(os.path.join(data_path, required_files[2]))
            eval_set = pd.read_csv(os.path.join(data_path, required_files[3]))

        return evidence_corpus, bi_encoder_training_set, cross_encoder_training_set, eval_set
