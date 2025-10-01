#!/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np


def create_arg_parser():
    p = argparse.ArgumentParser(description="Fine-tune a Transformer with Trainer")

    # Data / model arguments
    p.add_argument("--model_name", choices=["answerdotai/ModernBERT-base", "huawei-noah/TinyBERT_General_4L_312D", "distilbert-base-uncased", "bert-base-uncased"], default="bert-base-uncased")
    p.add_argument("--train_file", default="train.txt")
    p.add_argument("--dev_file", default="dev.txt")
    p.add_argument("--max_length", type=int, default=100)

    # Core TrainingArguments - note they have smart default settings
    p.add_argument("--output_dir", default="./out")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--eval_strategy", choices=["no", "steps", "epoch"], default="epoch")
    p.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default="no")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    return documents, labels


def compute_metrics(eval_pred):
    '''Metrics we compute for the dev set'''
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_micro": f1_micro, "f1_macro": f1_macro}


def print_results(metrics):
    '''Print final results for the dev set here'''
    acc = round(metrics["eval_accuracy"] * 100, 1)
    f1_micro = round(metrics["eval_f1_micro"] * 100, 1)
    f1_macro = round(metrics["eval_f1_macro"] * 100, 1)
    print (metrics)
    print(f"\n\nFinal metrics:\n")
    print(f"Accuracy: {acc}")
    print(f"Micro F1: {f1_micro}")
    print(f"Macro F1: {f1_macro}")


def prepare_data(args, tokenizer):
    '''Tokenize and build datasets + label encoder'''
    # First get the data
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    # Load labels the right way
    le = LabelEncoder()
    y_train_ids = le.fit_transform(Y_train).astype(np.int64)
    y_dev_ids = le.transform(Y_dev).astype(np.int64)

    # Tokenize the data
    # Note the settings for truncation and max length!
    tok_train = tokenizer(X_train, padding=True, truncation=True, max_length=args.max_length)
    tok_dev = tokenizer(X_dev, padding=True, truncation=True, max_length=args.max_length)

    # Make the library happy by using a dataset class
    train_ds = Dataset.from_dict({**tok_train, "labels": y_train_ids}).with_format("torch")
    dev_ds = Dataset.from_dict({**tok_dev, "labels": y_dev_ids}).with_format("torch")

    return train_ds, dev_ds, le


def main():
    '''Main function'''
    args = create_arg_parser()

    # Specify the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load and process the data
    train_ds, dev_ds, le = prepare_data(args, tokenizer)
    num_labels = len(le.classes_)

    # Setup the model, take model name from argument
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # Specify training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio if args.warmup_steps == 0 else 0.0,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        report_to=[])

    # Do the training!
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds, compute_metrics=compute_metrics)
    trainer.train()

    # Evaluate the training
    metrics = trainer.evaluate()
    print_results(metrics)

    # Potentially get back label to index mapping this way
    #print("Label index mapping:", {i: lab for i, lab in enumerate(le.classes_)})


if __name__ == "__main__":
    main()

