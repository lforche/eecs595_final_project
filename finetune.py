import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda")

class SimpleDataCollator:
    def __init__(self, tokenizer, max_length=None, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        labels = [feature['labels'] for feature in features]
        batch_size = len(features)

        # Create a list of dictionaries, each containing the relevant tensor
        inputs = {
            'input_ids': torch.cat([ex['input_ids'] for ex in features], dim=0),
            'token_type_ids': torch.cat([ex['token_type_ids'] for ex in features], dim=0),
            'attention_mask': torch.cat([ex['attention_mask'] for ex in features], dim=0),
            'labels': torch.tensor(labels, dtype=torch.int64).to(device),
        }

        return inputs

def load_data(tokenizer, params):
    label_mapping = {"entailment": 0, "contradiction": 1, "neutral": 2}

    def tokenize_function(data):
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        labels = label_mapping[data['label1']]

        tokenized_example = tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )

        tokenized_example["labels"] = torch.tensor(labels, dtype=torch.int64).to(device)
        return tokenized_example

    data_collator = SimpleDataCollator(
        tokenizer,
        max_length=params.max_length,
        pad_to_multiple_of=params.pad_to_multiple_of
    )

    train_data = pd.read_csv(params.train_dataset, delimiter='\t')  
    validation_data = pd.read_csv(params.validation_dataset, delimiter='\t')  
    test_data = pd.read_csv(params.test_dataset, delimiter='\t')  

    train_tokenized_datasets = train_data.apply(tokenize_function, axis=1)
    validation_tokenized_datasets = validation_data.apply(tokenize_function, axis=1)
    test_tokenized_datasets = test_data.apply(tokenize_function, axis=1)
    
    train_dataloader = DataLoader(train_tokenized_datasets, batch_size=params.batch_size, collate_fn=data_collator)
    validation_dataloader = DataLoader(validation_tokenized_datasets, batch_size=params.batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_tokenized_datasets, batch_size=params.batch_size, collate_fn=data_collator)

    return train_dataloader, validation_dataloader, test_dataloader

def finetune(model, train_dataloader, eval_dataloader, params):
    num_training_steps = params.num_epochs * len(train_dataloader)
    num_warmup_steps = 0
    lr = 5e-5
    optimizer = AdamW(model.parameters(), lr=lr)
    
    progress_bar = tqdm(range(num_training_steps))
    metric = load_metric('accuracy')
    
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    for epoch in range(params.num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        
        score = metric.compute()
        print('Validation Accuracy:', score['accuracy'])

    return model

def test(model, test_dataloader, prediction_save='predictions.torch'):
    metric = load_metric('accuracy')
    model.eval()
    all_predictions = []

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(list(predictions))
        metric.add_batch(predictions=predictions, references=batch["labels"])

    score = metric.compute()
    print('Test Accuracy:', score)
    torch.save(all_predictions, prediction_save)

def main(params):
    tokenizer = AutoTokenizer.from_pretrained(params.model)
    train_dataloader, validation_dataloader, test_dataloader = load_data(tokenizer, params)

    model = AutoModelForSequenceClassification.from_pretrained(params.model, num_labels=3)
    model.to(device)
    model = finetune(model, train_dataloader, validation_dataloader, params)

    test(model, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--train_dataset", type=str, default="train.tsv")
    parser.add_argument("--validation_dataset", type=str, default="valid.tsv")
    parser.add_argument("--test_dataset", type=str, default="test.tsv")
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser. add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--pad_to_multiple_of", type=int, default=None)

    params, unknown = parser.parse_known_args()
    main(params)