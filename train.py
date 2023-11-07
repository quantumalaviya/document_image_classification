import lightning.pytorch as pl
import torchmetrics
from tqdm import tqdm
from transformers import AdamW


class DocumentClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # hardcoding classes
        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=16
        )
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=16
        )

    def training_step(self, batch):
        # this is where the method_columns mapping from run.py
        # comes in handy, each model get only the inputs it needs
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True)

        preds = outputs.logits.argmax(-1)
        self.train_accuracy(preds, batch["labels"])
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_accuracy, prog_bar=True)

    def validation_step(self, batch):
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log("val_loss", loss, prog_bar=True)

        preds = outputs.logits.argmax(-1)
        self.val_accuracy(preds, batch["labels"])
        return loss

    def on_val_epoch_end(self):
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-4)
        return optimizer
