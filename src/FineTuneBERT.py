import argparse
import datetime
import itertools
import os
import random
import time
from collections import defaultdict
from enum import Enum
from pprint import pprint
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, \
    get_linear_schedule_with_warmup
from transformers.utils import logging


class Task(Enum):
    Train = 'Train'
    Eval = 'Eval'
    Infer = 'Infer'

    def __str__(self):
        return self.value


class Mode(Enum):
    Classification = 'Classification'
    Regression = 'Regression'

    def __str__(self):
        return self.value


class BERTTuner:

    def __init__(self, *,
                 base_model: str = 'google/bert_uncased_L-12_H-768_A-12',
                 mode: Mode = Mode.Regression,
                 input_model_path: str = None,
                 checkpoint_path: str = None,
                 output_path: str = None):
        self.mode = mode
        self.max_length = 100
        self.train_batch_size = 65
        self.infer_batch_size = 256
        self.train_validation_split = 0.9
        # Number of training epochs. The BERT authors recommend between 2 and 4.
        self.num_epochs = 6
        self.learning_rate = 2e-5
        self.warmup_proportion = 0.1
        self.output_path = output_path
        pprint(vars(self), indent=2)

        # Set the seed value all over the place to make training reproducible.
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)

        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            self.device = torch.device("cuda")
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        print(f'Loading BERT tokenizer with model {base_model}')
        logging.set_verbosity_error()
        self.tokenizer = BertTokenizer.from_pretrained(base_model, do_lower_case=True)

        if input_model_path:
            print(f'Loading BERT model from {input_model_path}')
            self.model = torch.load(input_model_path)
        else:
            print(f'Loading BERT model from {base_model}')
            self.model = BertForSequenceClassification.from_pretrained(
                base_model,
                num_labels=2 if self.mode is Mode.Classification else 1,
                # 2 for binary classification, 1 for regressions.
                output_attentions=False,
                output_hidden_states=False
            )

        self.model.to(self.device)  # might be redundant

        # Load AdamW from huggingface
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.learning_rate,
                               betas=(0.9, 0.999),  # default
                               eps=1e-6,  # default
                               weight_decay=0,  # default
                               correct_bias=True  # default
                               )

        if checkpoint_path is not None:
            print(f'Loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
        else:
            self.start_epoch = 0
            self.loss = 0

    def train(self, dataset_path: str) -> None:
        sentences, glosses, labels, targets2labels = self._load_dataset(dataset_path)
        dataset = self._prepare_input_data(sentences, glosses, labels)
        (training, validation, scheduler) = self._setup_training(dataset)
        self._train(training, validation, scheduler)

    def eval(self, dataset_path: str) -> None:
        sentences, glosses, labels, targets2labels = self._load_dataset(dataset_path)
        dataset = self._prepare_input_data(sentences, glosses, labels)
        self._eval(dataset, targets2labels)

    def infer(self, *, sentences: List[str], glosses: List[str]) -> List[float]:
        dataset = self._prepare_input_data(sentences, glosses)
        return self._infer(dataset)

    def _load_dataset(self, dataset_path: str) -> (List[str], List[str], List[Union[float, int]], Dict[str, List[Union[float, int]]]):
        # Load dataset
        print(f'Loading dataset from {dataset_path}')
        dataset = pd.read_csv(dataset_path, sep='\t', header=0, na_filter=False)
        print('Number of sentences in dataset: {:,}\n'.format(dataset.shape[0]))

        sentences = dataset.sentence.values.tolist()
        glosses = dataset.gloss.values.tolist()
        labels = [int(i) if self.mode is Mode.Classification else float(i) for i in dataset.label.values]
        targets = dataset.target_id.values.tolist()
        targets2labels = defaultdict(list)
        for i, t in enumerate(targets):
            targets2labels[t].append(i)

        assert len(sentences) == len(glosses) == len(labels)
        return sentences, glosses, labels, targets2labels

    def _prepare_input_data(self, sentences: List[str], glosses: List[str], labels: List[Union[float, int]] = None) -> TensorDataset:
        all_token_ids = []
        all_attention_masks = []
        all_segment_ids = []

        # Tokenize and encode texts
        size = len(sentences)
        # print(f'Encoding {str(size)} sentences as BERT input')
        for (sent, gloss) in tqdm(itertools.islice(zip(sentences, glosses), size), desc='Preparing input'):
            # if index % 100000 == 0:
            #    print(f'...{index} examples encoded in {time.time() - start_time}')

            assert sent and gloss

            # Tokenize texts
            sent_tokens = self.tokenizer.tokenize(sent)
            gloss_tokens = self.tokenizer.tokenize(gloss)

            # Truncate to max_length
            self._truncate_seq_pair(sent_tokens, gloss_tokens, self.max_length - 3)  # 3 -> CLS + SEP X 2

            # Add special chars
            tokens = ["[CLS]"] + sent_tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens += gloss_tokens + ["[SEP]"]
            segment_ids += [1] * (len(gloss_tokens) + 1)

            # Encode tokens
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # Create attention mask with 1s for real tokens and 0s for padding.
            mask = [1] * len(token_ids)

            # Add padding up to the sequence length.
            padding = [0] * (self.max_length - len(token_ids))
            token_ids += padding
            mask += padding
            segment_ids += padding

            assert len(token_ids) == self.max_length
            assert len(mask) == self.max_length
            assert len(segment_ids) == self.max_length

            all_token_ids.append(token_ids)
            all_attention_masks.append(mask)
            all_segment_ids.append(segment_ids)

        # convert to tensors
        all_token_ids = torch.tensor(all_token_ids, dtype=torch.long)
        all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

        if labels:
            if self.mode is Mode.Classification:
                labels = torch.tensor(labels[:size], dtype=torch.long)
            else:  # if self.mode is Mode.Regression:
                labels = torch.tensor(labels[:size], dtype=torch.float)

            return TensorDataset(all_token_ids, all_attention_masks, all_segment_ids, labels)
        else:
            return TensorDataset(all_token_ids, all_attention_masks, all_segment_ids)

    def _setup_training(self, dataset: TensorDataset) -> (DataLoader, DataLoader, LambdaLR):
        # Create a train-validation split
        train_size = int(self.train_validation_split * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=self.train_batch_size  # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=self.train_batch_size  # Evaluate with this batch size.
        )

        # Total number of training steps is [number of batches] x [number of epochs].
        # DataLoader.__len__  returns number of batches
        num_training_steps = len(train_dataloader) * self.num_epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=num_training_steps * self.warmup_proportion,
                                                    num_training_steps=num_training_steps)  # total including warm-up

        return train_dataloader, validation_dataloader, scheduler

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length: int) -> None:
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        # Taken from glossBERT -> https://github.com/HSLCY/GlossBERT
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    @staticmethod
    def _format_time(elapsed):
        # Round to the nearest second.
        elapsed_rounded = int(round(elapsed))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def _train(self, train_dataloader: DataLoader, validation_dataloader: DataLoader, scheduler: LambdaLR):
        self.model.cuda()

        total_t0 = time.time()
        training_stats = []

        for epoch_i in range(self.start_epoch, self.start_epoch + self.num_epochs):
            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.num_epochs))
            print('Training...')

            t0 = time.time()
            new_loss = self._train_pass(train_dataloader, scheduler)
            training_time = self._format_time(time.time() - t0)
            print("")
            print("  Average training loss: {0:.2f}, down from {0:.2f}".format(new_loss, self.loss))
            print("  Training epoch took: {:}".format(training_time))
            self.loss = new_loss

            # Save checkpoint
            path = self.output_path + '/semCor_' + str(self.mode) + '-' + str(self.max_length) + '-' + str(epoch_i + 1) + 'e' + '.pt'
            torch.save({
                'epoch': self.num_epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
            }, path)

            t0 = time.time()
            avg_val_accuracy, avg_val_loss = self._validate_pass(validation_dataloader)
            validation_time = self._format_time(time.time() - t0)
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': new_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            })

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self._format_time(time.time() - total_t0)))

        # Save whole model
        self.model.eval()
        path = self.output_path + '/semCor_' + str(self.mode) + '-' + str(self.max_length) + '.pt'
        torch.save(self.model, path)
        print(f'Model saved to {path}')

    def _train_pass(self, train_dataloader: DataLoader, scheduler: LambdaLR) -> float:
        # Reset the total loss for this epoch.
        total_train_loss = 0
        t0 = time.time()

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        self.model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = self._format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Copy tensors to GPU and unpack training batch from dataloader
            batch = tuple(t.to(self.device) for t in batch)
            token_ids, attention_mask, segment_ids, label_ids = batch

            # Always clear any previously calculated gradients before performing a backward pass.
            # Safer to do this rather than optimizer.zero_grad(), e.g. if there's multiple optimizers for one model
            self.model.zero_grad()

            # Perform a forward pass. token_type_ids are set to segment_ids to differentiate between sentence and gloss
            # It returns the "logits" --the model outputs prior to activation.
            logits = self.model(token_ids,
                                token_type_ids=segment_ids,
                                attention_mask=attention_mask,
                                labels=None)[0]  # loss is calculated separately, so no need to pass labels

            if self.mode == Mode.Classification:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))
            else:  # self.mode == Mode.Regression:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end.
            # `loss` is a Tensor containing a single value
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are modified based on their gradients,
            # the learning rate, etc.
            self.optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        return avg_train_loss

    def _validate_pass(self, validation_dataloader: DataLoader) -> (float, float):
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        self.model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Copy tensors to GPU and unpack training batch from dataloader
            batch = tuple(t.to(self.device) for t in batch)
            token_ids, attention_mask, segment_ids, label_ids = batch

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Perform a forward pass. token_type_ids are set to segment_ids to differentiate between sentence and
                # gloss It returns the "logits" --the model outputs prior to activation.
                logits = self.model(token_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=attention_mask,
                                    labels=None)[0]  # loss is calculated separately, so no need to pass labels

            if self.mode == Mode.Classification:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))
            elif self.mode == Mode.Regression:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
            else:
                raise KeyError(self.mode)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits_ = F.softmax(logits, dim=-1)
            logits_ = logits_.detach().cpu().numpy()
            label_ids_ = label_ids.to('cpu').numpy()
            outputs = np.argmax(logits_, axis=1)

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            tmp_eval_accuracy = np.sum(outputs == label_ids_)
            total_eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += token_ids.size(0)
            nb_eval_steps += 1

        total_eval_loss = total_eval_loss / nb_eval_steps
        total_eval_accuracy = total_eval_accuracy / nb_eval_examples

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        return avg_val_accuracy, avg_val_loss

    def _eval(self, dataset: TensorDataset, targets2labels: Dict[str, List[Union[float, int]]]) -> None:
        eval_dataloader = DataLoader(dataset, batch_size=self.infer_batch_size, shuffle=False)

        self.model.eval()
        logits = []

        with open(os.path.join(self.output_path, "results.txt"), "w") as f:
            for input_ids, input_mask, segment_ids, gold_labels in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)

                with torch.no_grad():
                    logits_batch = \
                        self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                   labels=None)[0]

                logits.append(logits_batch.detach().cpu())

            # flatten along outer dimension
            logits = torch.cat(logits, dim=0)

            # Calculate loss
            gold_labels = dataset.tensors[3]
            if self.mode is Mode.Classification:
                loss_fct = CrossEntropyLoss()
                eval_loss = loss_fct(logits.view(-1, 2), gold_labels)
            elif self.mode is Mode.Regression:
                loss_fct = MSELoss()
                eval_loss = loss_fct(logits.view(-1), gold_labels)

            # Calculate accuracy
            gold_picks = [max(candidates, key=lambda x: gold_labels[x]) for candidates in targets2labels.values()]

            if self.mode is Mode.Classification:
                probs = F.softmax(logits, dim=-1).numpy()
                pred_probs = probs[..., 1]  # probabilities of second label '1' -> of selecting each candidate sense
                pred_binary = np.argmax(probs, axis=1)  # preserves shape of second axis (column) -> maxes over rows
                # pred_picks = [index for index, candidate in enumerate(pred_binary) if candidate]  # classifier picks
                pred_picks = [max(candidates, key=lambda x: pred_probs[x]) for candidates in
                              targets2labels.values()]  # picks using score of '1' label

            elif self.mode is Mode.Regression:
                pred_values = logits.numpy().ravel()
                pred_picks = [max(candidates, key=lambda x: pred_values[x]) for candidates in targets2labels.values()]
                pred_binary = [1 if index in pred_picks else 0 for index in range(len(pred_values))]

            for i in range(len(pred_binary)):
                f.write(str(pred_binary[i]))
                for score in logits[i]:
                    f.write(" " + str(score.item()))
                f.write("\n")

        true_positives = set(pred_picks) & set(gold_picks)
        false_positives = set(pred_picks) - set(gold_picks)
        false_negatives = set(gold_picks) - set(pred_picks)
        accuracy = len(true_positives) / len(pred_picks)
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        f1 = (precision + recall) / 2.0

        print("***** Eval results *****")
        print(f"loss = {str(eval_loss.item())}")
        print(f"accuracy = {str(accuracy)}")
        print(f"precision = {str(precision)}")
        print(f"recall = {str(recall)}")
        print(f"f1 = {str(f1)}")

    def _infer(self, dataset: TensorDataset) -> List[float]:
        loader = DataLoader(dataset, batch_size=self.infer_batch_size, shuffle=False)
        self.model.eval()
        logits = []

        for input_ids, input_mask, segment_ids in tqdm(loader, desc="Inferring"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits_batch = \
                    self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                               labels=None)[0]

            logits.append(logits_batch.detach().cpu())

        # flatten along outer dimension
        logits = torch.cat(logits, dim=0)

        if self.mode is Mode.Classification:
            probs = F.softmax(logits, dim=-1).numpy()
            return probs[..., 1].tolist()  # probabilities of second label '1' -> of selecting each candidate sense
        elif self.mode is Mode.Regression:
            return logits.numpy().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate or infer with gloss-tuned BERT models")
    parser.add_argument(
        "-t",
        "--task",
        type=Task,
        choices=list(Task),
        default=Task.Infer,
        help="Task to address",
    )
    parser.add_argument(
        "-b",
        "--base_model",
        type=str,
        default='google/bert_uncased_L-12_H-768_A-12',
        help="Specify name of transformers base model",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=Mode,
        choices=list(Mode),
        default=Mode.Classification,
        help="Output mode of the neural model",
    )
    parser.add_argument(
        "-i",
        "--input_model",
        type=str,
        default='/home/gerard/data/GlossBERT_datasets/semCor_cased_Mode.Classification-100-6e.pt',
        help="Specify path to input model file",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default='/home/gerard/data/GlossBERT_datasets/semCor_cased_Mode.Classification-100-6e.pt',
        help="Specify path to checkpoint file",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default='',
        help="Specify path to dataset file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default='/home/gerard/data/GlossBERT_datasets',
        help="Specify path to output folder",
    )
    cml_args = parser.parse_args()

    if cml_args.task == Task.Train:
        tuner = BERTTuner(
            base_model=cml_args.base_model,
            mode=Mode[cml_args.mode],
            output_path=cml_args.output)
        tuner.train(dataset_path=cml_args.dataset)
    elif cml_args.task == Task.Eval:
        tuner = BERTTuner(
            base_model=cml_args.base_model,
            mode=cml_args.mode,
            checkpoint_path=cml_args.checkpoint,
            output_path=cml_args.output)
        tuner.eval(dataset_path=cml_args.dataset)

    # Train
    # tuner = BERTTuner(
    #     dataset_path='/home/gerard/data/GlossBERT_datasets/Training_Corpora/SemCor/semcor_train_sent_cls.csv',
    #     output_path='/home/gerard/data/GlossBERT_datasets',
    #     mode=Mode['Classification'])
    # tuner.run(train=True)

    # Eval classification
    # tuner = BERTTuner(
    #     dataset_path='/home/gerard/data/GlossBERT_datasets/Evaluation_Datasets/senseval3/senseval3_test_sent_cls.csv',
    #     output_path='/home/gerard/data/GlossBERT_datasets/',
    #     input_model_path='/home/gerard/data/GlossBERT_datasets/SemCor_full.pt',
    #     checkpoint_path='/home/gerard/data/GlossBERT_datasets/Sent_CLS_WS/pytorch_model.bin',
    #     mode=Mode.Classification)

    # Eval regression
    # tuner = BERTTuner(
    #     # dataset_path='/home/gerard/data/GlossBERT_datasets/Evaluation_Datasets/senseval3/senseval3_test_sent_cls.csv',
    #     # output_path='/home/gerard/data/GlossBERT_datasets/',
    #     # input_model='/home/gerard/data/GlossBERT_datasets/SemCor_Mode.Regression-4e.pt',
    #     # mode=Mode.Regression)
    # tuner.run(train=False, output_path='/home/gerard/data/GlossBERT_datasets/')
