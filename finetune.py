import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW, SGD
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric
from datasets import concatenate_datasets
from sklearn.model_selection import train_test_split

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
        # Ensure that 'input_ids', 'attention_mask', and 'labels' are present in each feature
        inputs = {}
        for key in ['input_ids', 'attention_mask', 'labels']:
            if any(key not in feature for feature in features):
                raise ValueError(f"Key '{key}' is missing in one or more features.")

        # Stack 'input_ids' and 'attention_mask'
        inputs['input_ids'] = torch.stack([feature['input_ids'] for feature in features])
        inputs['attention_mask'] = torch.stack([feature['attention_mask'] for feature in features])
        inputs['labels'] = torch.tensor([feature['labels'] for feature in features], dtype=torch.int64).to(device)

        return inputs

def load_data(tokenizer, params):
    label_ints = { "true": 1, "false": 0}
    
    def tokenize_function(data):
        dialog_text_list = data['dialog_text_list']
        speakers = data['dialog_speaker_list']
        h = data['h']
        labels = data['entailment']

        all_inputs = []
        curr_inputs = ""

        for i in range(len(dialog_text_list)):
            curr_inputs += "[CLS] "
            curr_inputs += str(speakers[i])[1:-1]
            curr_inputs += " [SEP] "
            curr_inputs += str(dialog_text_list[i])[1:-1]
            curr_inputs += " [SEP] "
            curr_inputs += str(h[i])
            curr_inputs += " [SEP] "
            all_inputs.append(curr_inputs)
            curr_inputs = ""

        tokenized_example = tokenizer(
            all_inputs,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )

        tokenized_example["labels"] = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)
        print(all_inputs[1])
        print(labels[1])
        
        return tokenized_example

    combined_data = load_dataset('csv', data_files='random_training_data.csv')['train']

    # Split into training, validation, and test
    train_indices, validation_test_indices = train_test_split(range(len(combined_data)), test_size=0.2, random_state=42, shuffle=True)

    # Use integer indexing for the split
    train_data = combined_data.select(train_indices)
    validation_test_data = combined_data.select(validation_test_indices)

    # Further split the combined data into validation and test sets
    validation_indices, test_indices = train_test_split(range(len(validation_test_data)), test_size=0.2, random_state=42)
    validation_data = validation_test_data.select(validation_indices)
    test_data = validation_test_data.select(test_indices)
    accepted_keys = ["input_ids", "attention_mask", "labels"]


    # Set format to "torch"
    train_data = train_data.map(tokenize_function, batched=True)
    train_data.set_format("torch")
    for key in train_data.features.keys():
        if key not in accepted_keys:
            train_data = train_data.remove_columns(key)

    validation_data = validation_data.map(tokenize_function, batched=True)
    validation_data.set_format("torch")
    for key in validation_data.features.keys():
        if key not in accepted_keys:
            validation_data = validation_data.remove_columns(key)

    test_data = test_data.map(tokenize_function, batched=True)
    test_data.set_format("torch")
    for key in test_data.features.keys():
        if key not in accepted_keys:
            test_data = test_data.remove_columns(key)

    train_dataloader = DataLoader(train_data)
    validation_dataloader = DataLoader(validation_data)
    test_dataloader = DataLoader(test_data)

    return train_dataloader, validation_dataloader, test_dataloader

def finetune(model, train_dataloader, eval_dataloader, params):
    lr = 1e-4
    device = next(model.parameters()).device

    # Define optimizer
    optimizer = SGD(model.parameters(), lr=lr)

    # Define loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Define learning rate scheduler
    num_training_steps = params.num_epochs * len(train_dataloader)
    num_warmup_steps = 5
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    # Use tqdm for a progress bar
    progress_bar = tqdm(range(num_training_steps))
    
    # Define metric for evaluation
    metric = load_metric('accuracy')
    count = 0
    all_outputs = []
    all_labels = []
    for epoch in range(params.num_epochs):
        model.train()  # set model to training mode
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}  # collect data to be inputted
            outputs = model(**batch)  # run model on data
            logits = outputs.logits
            labels = batch['labels'].float().squeeze(1)  # Assuming labels are already in the batch
            loss = loss_fn(logits, labels)  # calculate loss
            loss.backward()  # backpropagation

            optimizer.step()  # update model parameters
            optimizer.zero_grad()  # zero out the gradients
            lr_scheduler.step()  # update the learning rate (linear decay)
            progress_bar.update(1)

            all_outputs.append(logits.item())
            all_labels.append(labels.item())

        count = 0
        all_outputs = []
        all_labels = []
        # Evaluation
        model.eval()
        for eval_batch in eval_dataloader:
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}  # collect data to be inputted
            with torch.no_grad():
                eval_outputs = model(**eval_batch)
                eval_logits = eval_outputs.logits
                eval_labels = eval_batch['labels'].float().squeeze(1)
                eval_loss = loss_fn(eval_logits, eval_labels)
                metric.add_batch(predictions=(eval_logits > 0).long(), references=eval_labels.long())
            all_outputs.append(eval_logits.item())
            all_labels.append(eval_labels.item())

        eval_score = metric.compute()
        print(f'Epoch {epoch + 1}/{params.num_epochs}, Eval Accuracy: {eval_score["accuracy"]:.4f}')

    return model

def test(model, test_dataloader, prediction_save='predictions.torch'):
    metric = load_metric('accuracy')
    model.eval()
    all_predictions = []

    count = 0
    all_outputs = []
    all_labels = []
    threshold = 0
    # Evaluation
    model.eval()
    for eval_batch in test_dataloader:
        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}  # collect data to be inputted
        with torch.no_grad():
            eval_outputs = model(**eval_batch)
            eval_logits = eval_outputs.logits
            eval_labels = eval_batch['labels'].float().squeeze(1)
    
        all_outputs.append(eval_logits.item())
        all_labels.append(eval_labels.item())

    optimal_threshold = get_optimal_threshold(all_outputs, all_labels)
    
    for i in range(len(all_labels)):
        if all_outputs[i] > optimal_threshold:
            all_predictions.append(1.0)
        else:
            all_predictions.append(0.0)

    for i in range(len(all_labels)):
        metric.add(predictions=all_predictions[i], references=all_labels[i])

    score = metric.compute() #compare predictions to references
    print('Test Accuracy:', score)
    torch.save(all_predictions, prediction_save) #save predictions

def checksum(model):
    s = 0.0
    for param in model.parameters():
        s += torch.sum(param)
    return s

def print_predictions(predictions):
    num_true = 0
    num_false = 0
    for i in predictions:
        if i == 1:
            num_true += 1
        else:
            num_false += 1
    
    print("True : ", num_true)
    print("False: ", num_false)

def get_optimal_threshold(all_logits, all_labels):
    import numpy as np
    from sklearn.metrics import accuracy_score

    # Find the threshold that maximizes accuracy
    thresholds = np.unique(all_logits)
    accuracies = [accuracy_score(all_labels, (all_logits > threshold).astype(int)) for threshold in thresholds]
    optimal_threshold = thresholds[np.argmax(accuracies)]
    print(f'Optimal Threshold for Accuracy: {optimal_threshold}')

    return optimal_threshold


def main(params):
    tokenizer = AutoTokenizer.from_pretrained(params.model)
    train_dataloader, validation_dataloader, test_dataloader = load_data(tokenizer, params)

    model = AutoModelForSequenceClassification.from_pretrained(params.model, num_labels=1)
    model.classifier = torch.nn.Linear(model.config.hidden_size, out_features=1)
    model.classifier.bias.data = torch.tensor([0.0], dtype=torch.float)
    model.classifier.weight.data = torch.tensor([[1.0] * model.config.hidden_size], dtype=torch.float)
    model.to(device)

    model = finetune(model, train_dataloader, validation_dataloader, params)

    test(model, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")
    # parser.add_argument("--model", type=str, default="xlnet-base-cased")
    # parser.add_argument("--model", type=str, default="bert-base-uncased")
    # parser.add_argument("--model", type=str, default="albert-base-v2")
    # parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--model", type=str, default="microsoft/deberta-base")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--pad_to_multiple_of", type=int, default=None)

    params, unknown = parser.parse_known_args()
    main(params)