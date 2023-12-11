import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

def get_writer(output_directory, log_directory):

    logging_path=f'{output_directory}/{log_directory}'
    if os.path.exists(logging_path) == False:
        os.makedirs(logging_path)
    writer = CustomWriter(logging_path)
            
    return writer

class CustomWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(CustomWriter, self).__init__(log_dir)
        self.log_data = {'name': [], 'phase': [], 'loss': [], 'global_step': []}
        
    def add_losses(self, name, phase, loss, global_step):
        self.add_scalar(f'{name}/{phase}', loss, global_step)
    
    def save_to_csv(self, name:str, phase:str, loss, global_step):
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().detach().numpy()
        if isinstance(global_step, torch.Tensor):
            global_step = global_step.cpu().detach().numpy()
        
        self.log_data['name'].append(name)
        self.log_data['phase'].append(phase)
        self.log_data['loss'].append(loss)
        self.log_data['global_step'].append(global_step)
        
        df = pd.DataFrame(self.log_data)
        csv_path = os.path.join(self.log_dir, 'log_data.csv')
        df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
        

def save_checkpoint(model, optimizer, learning_rate, epoch, filepath):
    print(f"Saving model and optimizer state at iteration {epoch} to {filepath}")
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{epoch}')
    
