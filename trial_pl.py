import os
import glob
import argparse
from icecream import ic
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import lightning as L
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from util.utils import get_writer, save_checkpoint
from util.cal_pearson import l1_loss, pearson_loss, pearson_metric
from util.dataset import RegressionDataset
from models.FFT_block import Decoder


parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--win_len", type=int, default=10)
parser.add_argument("--sample_rate", type=int, default=64)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--g_con", default=False, help="experiment for within subject")

parser.add_argument(
    "--in_channel", type=int, default=64, help="channel of the input eeg signal"
)
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--d_inner", type=int, default=1024)
parser.add_argument("--n_head", type=int, default=2)
parser.add_argument("--n_layers", type=int, default=8)
parser.add_argument("--fft_conv1d_kernel", type=tuple, default=(9, 1))
parser.add_argument("--fft_conv1d_padding", type=tuple, default=(4, 0))
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--lamda", type=float, default=0.2)
parser.add_argument("--writing_interval", type=int, default=10)
parser.add_argument("--saving_interval", type=int, default=10)

parser.add_argument(
    "--dataset_folder",
    type=str,
    default="/home/kunal/eeg_data/derivatives/",
    help="write down your absolute path of dataset folder",
)
parser.add_argument("--split_folder", type=str, default="downsample")
parser.add_argument(
    "--experiment_folder", default="1", help="write down experiment name"
)

args = parser.parse_args()

# Set the parameters and device.

input_length = args.sample_rate * args.win_len


class DecoderPL(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = Decoder(**vars(args))
        self.lamda = 0.2

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ic("Training started")
        # training_step defines the train loop.
        # it is independent of forward
        inputs, labels = batch
        outputs = self.model(inputs)
        ic(outputs.shape)
        # Loss for the train in epochs
        l_p = pearson_loss(labels, outputs)
        l_1 = l1_loss(outputs, labels)
        # loss = l_p + self.lamda * l_1
        loss = l_p
        ic(l_p.shape)
        ic(l_1.shape)
        loss = loss.mean()
        # Logging to TensorBoard (if installed) by default
        # self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        ic(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ic("Validation started")
        inputs, labels = batch
        inputs = inputs.squeeze(0)
        labels = labels.squeeze(0)
        outputs = self.model(inputs)
        loss = pearson_loss(outputs, labels).mean()
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        ic("Test started")
        inputs, labels = batch
        inputs = inputs.squeeze(0)
        labels = labels.squeeze(0)
        outputs = self.model(inputs)
        loss = pearson_loss(outputs, labels).mean()
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.0005,
            betas=(0.9, 0.98),
            eps=1e-09,
        )
        lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
        return [optimizer], [lr_scheduler]


class RegressionDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir"):
        super().__init__()
        self.data_dir = data_dir
        self.features = ["eeg", "mel"]

    def setup(self, stage: str):
        # self.mnist_test = MNIST(self.data_dir, train=False)
        # self.mnist_predict = MNIST(self.data_dir, train=False)
        # mnist_full = MNIST(self.data_dir, train=True)
        # self.mnist_train, self.mnist_val = random_split(
        #     mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        # )
        if stage == "fit":
            data_folder = os.path.join(self.data_dir, "downsample")
            train_files = [
                x
                for x in glob.glob(os.path.join(data_folder + "/train/", "train_-_*"))
                if os.path.basename(x).split("_-_")[-1].split(".")[0] in self.features
            ]
            self.train_set = RegressionDataset(
                files=train_files,
                input_length=320,
                channels=64,
                task="train",
                g_con=False,
            )
            data_folder = os.path.join(self.data_dir, "split_data")
            val_files = [
                x
                for x in glob.glob(os.path.join(data_folder + "/val/", "val_-_*"))
                if os.path.basename(x).split("_-_")[-1].split(".")[0] in self.features
            ]
            self.val_set = RegressionDataset(
                files=val_files, input_length=320, channels=64, task="val", g_con=False
            )
        if stage == "test":
            data_folder = os.path.join(self.data_dir, "split_data")
            test_files = [
                x
                for x in glob.glob(os.path.join(data_folder + "/test/", "test_-_*"))
                if os.path.basename(x).split("_-_")[-1].split(".")[0] in self.features
            ]
            self.test_set = RegressionDataset(
                files=test_files,
                input_length=320,
                channels=64,
                task="test",
                g_con=False,
            )
            pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=64,
            num_workers=4,
            drop_last=False,
            shuffle=True,
            pin_memory=True

        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=4,
            drop_last=False,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=4,
            drop_last=False,
            shuffle=False,
        )


decoder_pl = DecoderPL()
regression_dm = RegressionDataModule(data_dir="/home/kunal/eeg_data/derivatives/")
trainer = L.Trainer(
    fast_dev_run=True,
    max_epochs=1000,
    accelerator="gpu",
    devices=[2],
    logger=[CSVLogger(save_dir="lightning_logs/", name="experiment_hq_noearly_l1"), TensorBoardLogger(save_dir="lightning_logs/", name="experiment_hq_noearly_l1")],
    callbacks=[
        TQDMProgressBar(refresh_rate=10),
        # EarlyStopping(monitor="val/loss", patience=20, mode="min"),
        ModelCheckpoint(monitor='val/loss', mode='min', every_n_epochs=10, save_top_k=5)
    ],
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
)

trainer.fit(decoder_pl, regression_dm)
# trainer.test(decoder_pl, regression_dm)