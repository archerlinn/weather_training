import os
import torch
from torch.utils.data import Dataset, DataLoader
import nrrd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class WeatherDataset(Dataset):
    def __init__(self, data_files, transform=None):
        self.data_files = data_files
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data, _ = nrrd.read(self.data_files[idx])  # Load grid data from .nrrd file
        data = data[:, :, 0]  # Assuming we want the first channel

        # Normalize angular velocity data without resizing
        data_min, data_max = np.min(data), np.max(data)
        data = (data - data_min) / (data_max - data_min)  # Normalize to 0-1 range
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform:
            data = self.transform(data)

        target = data
        return data, target

# Training Function with TensorBoard logging
def train_model(model, dataloader, num_epochs=10000, save_path='/content/drive/MyDrive/Weather_Model/model_weights_epoch_latest.pth'):
    print("Training the model...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loss_history = []  # To store loss values for plotting

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs, _ = model(inputs)  # Unpack only `outputs`
            outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)  # Store loss for this epoch

        # Log average loss to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Save the model weights as "model_weights_epoch_latest.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss}")
        print(f"Model weights saved at {save_path}")

    print("Training complete!")
    writer.close()  # Close the TensorBoard writer

# Visualization Function for Bottleneck
def visualize_bottleneck(dataloader, model):
    model.eval()
    bottleneck_representations = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs, bottleneck = model(inputs)  # Ensure the model's forward method returns both
            bottleneck_representations.append(bottleneck.cpu().numpy())
    bottleneck_representations = np.concatenate(bottleneck_representations, axis=0).reshape(-1, 128)
    pca = PCA(n_components=2)
    bottleneck_2d = pca.fit_transform(bottleneck_representations)
    plt.scatter(bottleneck_2d[:, 0], bottleneck_2d[:, 1], alpha=0.6, c='blue')
    plt.title("2D PCA of Bottleneck Representations")
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Mount Google Drive
    
    file_dir = '/content/drive/MyDrive/Weather_Model/CFD_Subset'
    data_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith(".nrrd")]
    dataset = WeatherDataset(data_files=data_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet(in_channels=1, out_channels=1)  # Initialize U-Net model
    model = model.to('cuda')  # Move the model to GPU if available

    # Initialize the optimizer after the model is created
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load the latest checkpoint if available
    checkpoint_path = '/content/drive/MyDrive/Weather_Model/model_weights_epoch_latest.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False) 
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    else:
        print("No checkpoint found, starting training from scratch.")
        start_epoch = 1

    # Start training from the correct epoch
    train_model(model, dataloader, num_epochs=10000, save_path=checkpoint_path)  # Pass `start_epoch`
    visualize_bottleneck(dataloader, model)
