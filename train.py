import os
import glob
import argparse
from icecream import ic
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from util.utils import get_writer, save_checkpoint
from util.cal_pearson import l1_loss, pearson_loss, pearson_metric
from util.dataset import RegressionDataset
from models.FFT_block import Decoder

parser = argparse.ArgumentParser()

parser.add_argument('--epoch',type=int, default=1000)
parser.add_argument('--batch_size',type=int, default=64)
parser.add_argument('--win_len',type=int, default = 10)
parser.add_argument('--sample_rate',type=int, default = 64)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--g_con', default=True, help="experiment for within subject")

parser.add_argument('--in_channel', type=int, default=64, help="channel of the input eeg signal")
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_inner', type=int, default=1024) 
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_layers',type=int, default=8)
parser.add_argument('--fft_conv1d_kernel', type=tuple,default=(9, 1))
parser.add_argument('--fft_conv1d_padding',type=tuple, default= (4, 0))
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--dropout',type=float,default=0.3)
parser.add_argument('--lamda',type=float,default=0.2)
parser.add_argument('--writing_interval', type=int, default=10)
parser.add_argument('--saving_interval', type=int, default=10)

parser.add_argument('--dataset_folder',type= str, default="/home/kunal/eeg_data/derivatives/", help='write down your absolute path of dataset folder')
parser.add_argument('--split_folder',type= str, default="downsample")
parser.add_argument('--experiment_folder',default="1", help='write down experiment name')

args = parser.parse_args()

 # Set the parameters and device.

input_length = args.sample_rate * args.win_len 
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Provide the path of the dataset.
# which is split already to train, val, test (1:1:1).
data_folder = os.path.join(args.dataset_folder, args.split_folder)
features = ["eeg"] + ["mel"]
print(data_folder)

data_folder_2 = os.path.join(args.dataset_folder, "split_data")
# Create a directory to store (intermediate) results.
result_folder = 'test_results'
if args.experiment_folder is None:
    experiment_folder = "fft_nlayer{}_dmodel{}_nhead{}_win{}".format(args.n_layers, args.d_model, args.n_head, args.win_len)
else: experiment_folder = args.experiment_folder

save_path = os.path.join(result_folder, experiment_folder)
writer = get_writer(result_folder, experiment_folder)

def main():

    # Set the model and optimizer, scheduler.
    model = Decoder(**vars(args)).to(device)
    input_shape = ic(next(model.parameters()).size())
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate,
                                betas=(0.9, 0.98),
                                eps=1e-09)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    # Define train set and loader.
    train_files = [x for x in glob.glob(os.path.join(data_folder, "train", "train_-_*")) if os.path.basename(x).split("_-_")[-2].split(".")[0] in features]
    print(train_files[0])
    import numpy as np

    path = train_files[1]
    mel = np.load(path)
    print("Checking if data is here")
    print(mel.shape)
    train_set= RegressionDataset(train_files, 320, args.in_channel, 'train', args.g_con)
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size = args.batch_size,
            num_workers = 4,
            sampler = None,
            drop_last=True,
            shuffle=True)
    # print("************************************************************************")
    # print("Length of train DataLoader:",len(train_dataloader))
    # print("Number of train files: ", len(train_files))
    ic(len(train_files))
    ic(len(train_set))
    # ic(train_dataloader.size())

    # Define validation set and loader.
    val_files = [x for x in glob.glob(os.path.join(data_folder_2 +"/val/", "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    val_set = RegressionDataset(val_files, 320, 64, 'val', False)
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size = 1,
        num_workers = 4,
        sampler = None,
        drop_last=False,
        shuffle=False)
    ic(len(val_dataloader))
    ic(len(val_files))
    ic(len(val_set))

    # Define test set and loader.
    test_files = [x for x in glob.glob(os.path.join(data_folder_2 +"/test/", "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    test_set = RegressionDataset(test_files, 320, 64, 'test', False)
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size = 1,
        num_workers = 4,
        sampler = None,
        drop_last=False,
        shuffle=False)
    ic(len(test_dataloader))
    ic(len(test_files))
    ic(len(test_set))

    import numpy as np

    path = val_files[0]
    mel = np.load(path)
    print("Checking if data is here")
    print(mel.shape)
    # for inputs, labels in val_dataloader:
    #     print(inputs,labels)


    #Train the model.
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            # sub_id = sub_id.to(device)
            outputs = model(inputs)

            l_p = pearson_loss(outputs, labels) 
            l_1 = l1_loss(outputs, labels)
            loss = l_p + args.lamda * l_1
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        if epoch % args.writing_interval == 0:
            print(f'|-Train-|{epoch}: {train_loss:.3f}')
            writer.add_losses("Loss", "train",  train_loss, epoch)
            writer.add_losses("Loss_l1", "train",  train_loss, epoch)
            writer.save_to_csv("Loss", "train",  train_loss, epoch)

        # Validate the model.
        val_loss = 0
        val_metric = 0
        
        if epoch % args.writing_interval == 0:
            ic()
            model.eval()

            with torch.no_grad():
                ic()
                for val_inputs, val_labels in val_dataloader:
                    val_inputs = val_inputs.squeeze(0).to(device)
                    val_labels = val_labels.squeeze(0).to(device)
                    # val_sub_id = val_sub_id.to(device)
                    
                    val_outputs = model(val_inputs)
                    val_loss   += pearson_loss(val_outputs, val_labels).mean()
                    val_metric += pearson_metric(val_outputs, val_labels).mean()

                val_loss /= len(val_dataloader)
                val_metric /= len(val_dataloader)
                val_metric = val_metric.mean()

                print(f'|-Validation-|{epoch}: {val_loss.mean().item():.3f} {val_metric.item():.3f}')
                writer.add_losses("Loss", "Validation",  val_loss, epoch)
                writer.add_losses("Pearson", "Validation",  val_metric, epoch)
                writer.save_to_csv("Loss", "Validation",  val_loss, epoch)

                # Test the model.
                test_loss = 0
                test_metric = 0

                for test_inputs, test_labels in test_dataloader:
                    test_inputs = test_inputs.squeeze(0).to(device)
                    test_labels = test_labels.squeeze(0).to(device)
                    # test_sub_id = test_sub_id.to(device)

                    test_outputs = model(test_inputs)
                    test_loss += pearson_loss(test_outputs, test_labels).mean()
                    test_metric += pearson_metric(test_outputs, test_labels).mean()
            
                test_loss /= len(test_dataloader)
                test_metric /= len(test_dataloader)
                test_metric = test_metric.mean()    
                print(f'|-Test-|{epoch}: {test_loss.mean().item():.3f} {test_metric.item():.3f}')
                writer.add_losses("Loss", "Test",  test_loss.mean().item(), epoch)
                writer.add_losses("Pearson", "Test",  test_metric, epoch)
                writer.save_to_csv("Loss", "Test",  test_loss.mean().item(), epoch)

        if epoch % args.saving_interval == 0:
            learning_rate = print(optimizer.param_groups[0]["lr"])
            save_checkpoint(model, optimizer, learning_rate, epoch, save_path)    

        scheduler.step()


if __name__ == '__main__':
    main()
