import os

import lightning as pl
import torch
from datasets import Array2D, Array3D, ClassLabel, Features, Sequence, Value
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          LayoutLMv2FeatureExtractor, LayoutLMv2Processor)

from dataset import dataset_generator
from train import DocumentClassifier
from utils import encode_example, encode_example_v2

# define the method
# one of ["bert-base-uncased", "microsoft/layoutlm-base-uncased", ""microsoft/layoutlmv2-base-uncased""]
method = "microsoft/layoutlm-base-uncased"

# define whether to only test or also train
test_only = False

# checkpoint to test the model on
CKPT_PATH = None

# define the batch_size
batch_size = 8

# define train epochs
epochs = 10

# auto detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# infos paths
train_infos_path = "./data/train_infos.pkl"
val_infos_path = "./data/val_infos.pkl"

# a brute force implementation to keep things simple
# we need to define the features ourselves so that mapping
# during encoding is possible.
features = {
    "input_ids": Sequence(feature=Value(dtype="int64")),
    "attention_mask": Sequence(Value(dtype="int64")),
    "token_type_ids": Sequence(Value(dtype="int64")),
    "labels": ClassLabel(names=list(map(str, range(0, 16)))),
    "bbox": Array2D(dtype="int64", shape=(512, 4)),
}

# very non-robust but enough for the scope of this project
# adding image as v2 uses visual features as well.
if "v2" in method:
    features["image"] = Array3D(dtype="int64", shape=(3, 224, 224))

# defines the dataset columns to propagate to the torch dataset from the hf dataset
method_columns = {
    "bert-base-uncased": ["input_ids", "attention_mask", "token_type_ids", "labels"],
    "microsoft/layoutlm-base-uncased": [
        "input_ids",
        "bbox",
        "attention_mask",
        "token_type_ids",
        "labels",
    ],
    "microsoft/layoutlmv2-base-uncased": [
        "image",
        "input_ids",
        "bbox",
        "attention_mask",
        "token_type_ids",
        "labels",
    ],
}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(method)

    # hardcoding num_labels
    model = AutoModelForSequenceClassification.from_pretrained(method, num_labels=16)

    # generate HF datasets
    if not test_only:
        train_ds = dataset_generator(train_infos_path)
    val_ds = dataset_generator(val_infos_path)

    # encode the OCR outputs and some additional preprocessing
    if "v2" in method:
        # OCR has been precomputed, setting it False saves a lot of time.
        feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
        processor = LayoutLMv2Processor(feature_extractor, tokenizer)

        if not test_only:
            train_ds = train_ds.map(
                lambda example: encode_example_v2(example, processor),
                features=Features(features),
                num_proc=os.cpu_count(),
            )
        val_ds = val_ds.map(
            lambda example: encode_example_v2(example, processor),
            features=Features(features),
            num_proc=os.cpu_count(),
        )
    else:
        # TODO: DRY
        if not test_only:
            train_ds = train_ds.map(
                lambda example: encode_example(example, tokenizer),
                features=Features(features),
                num_proc=os.cpu_count(),
            )
        val_ds = val_ds.map(
            lambda example: encode_example(example, tokenizer),
            features=Features(features),
            num_proc=os.cpu_count(),
        )

    # convert to a torch dataset and feed to a dataloader for training
    if not test_only:
        train_ds.set_format(type="torch", columns=method_columns[method])
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
        )
    val_ds.set_format(type="torch", columns=method_columns[method])
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
    )

    classifier = DocumentClassifier(model)

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath='checkpoints/', monitor="val_loss")
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback])

    if not test_only:
        trainer.fit(
            model=classifier,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    trainer.validate(
        model=classifier,
        dataloaders=val_loader,
        ckpt_path=CKPT_PATH if CKPT_PATH else "best",
    )
