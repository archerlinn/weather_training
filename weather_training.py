import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nrrd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from unet import UNet

class WeatherDataset(Dataset):
    def __init__(self, data_files, transform=None):
        self.data_files = data_files
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data, _ = nrrd.read(self.data_files[idx])
        data = data[:, :, 0]
        data_min, data_max = np.min(data), np.max(data)
        data = (data - data_min) / (data_max - data_min)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        target = data
        return data, target

def train_model(model, train_loader, val_loader, num_epochs=10000, save_path='model_weights_epoch_latest.pth'):
    print("Training the model...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loss_history = []
    val_loss_history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to('cuda'), val_targets.to('cuda')
                val_outputs, _ = model(val_inputs)
                val_outputs = F.interpolate(val_outputs, size=val_targets.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(val_outputs, val_targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        wandb.log({"train_loss": avg_loss, "val_loss": avg_val_loss})

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        }, save_path)
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_loss}, Val Loss: {avg_val_loss}")

    print("Training complete!")

# Main Execution
if __name__ == "__main__":
    wandb.init(project="weather_forecast")

    file_dir = '/content/drive/MyDrive/Weather_Model/CFD_Subset'
    data_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith(".nrrd")]
    dataset = WeatherDataset(data_files=data_files)

    # Split the dataset into 80% training and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=1, out_channels=1)
    model = model.to('cuda')

    # Load the latest checkpoint if available
    checkpoint_path = '/content/drive/MyDrive/Weather_Model/model_weights_epoch_latest.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("No checkpoint found, starting training from scratch.")
        start_epoch = 1

    train_model(model, train_loader, val_loader, num_epochs=10000, save_path=checkpoint_path)
