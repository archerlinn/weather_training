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
import matplotlib.pyplot as plt
from unet import UNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login()

class DataLoader(Dataset):
    def __init__(self, data_files, transform=None):
        self.data_files = data_files
        self.transform = transform
        #data normalization - compute mean and std
        data_list = []
        for data_files in self.data_files:
            # Load the data (assuming it's a NumPy array or similar)
            data = np.load(data_files)  # Modify if needed based on your data format
            data_list.append(data)
        # Convert the list of arrays to a single tensor
        data_tensor = torch.from_numpy(np.asarray(data_list).astype(np.float32))
        # Compute mean and standard deviation across all dimensions except the channel index (assumed to be index 2)
        self.mean = data_tensor.mean(dim=[0,2,3]])  # Exclude channel dimension (index 1)
        self.std = data_tensor.std(dim=[0,2,3])    # Exclude channel dimension (index 1)
        """
        list = []
        for i in range(len(data_file)):
            list.append(data_file[i])
        tensor = torch.from_numpy(np.asarray(list).astype(np.float32))
        self.mean = tensor.mean(dim=[everything except channel index (i.e the index that corresponds to 2)])
        self.std = similar to above
        """
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data, _ = nrrd.read(self.data_files[idx])
        #normalizing data using the global mean and std
        data = torch.from_numpy(data.astype(np.float32)).permute(1, 2, 0) #make sure channel is in the last before mean is computed i.e. change (2, 1800, 1420) to (1800, 1420, 2) using torch.permute
        data = (data - self.mean) / self.std
        #data = torch.tensor(data, dtype=torch.float32).unsqueeze(0) - I don't need this right?
        target = data
        return data, target

def plot_prediction_vs_ground_truth(prediction, ground_truth):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ground_truth.squeeze().cpu().numpy(), cmap='viridis')
    axes[0].set_title("Ground Truth")
    axes[1].imshow(prediction.squeeze().cpu().numpy(), cmap='viridis')
    axes[1].set_title("Prediction")
    plt.show()

def train_model(model, train_loader, val_loader, num_epochs, save_path):
    print("Training the model...")
    
    #Optimizer Updates the model’s parameters (weights) based on the gradients computed during back propagation.
    #Adam is Efficient (computationally inexpensive), Adaptive (handles noisy data well), and Good at handling sparse gradients.
    #lr = leaning rate (step size for updating the parameters), smaller = stable but slower
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.MSELoss() #We use MSELoss for regression to predict continuous value
    loss_history = []
    val_loss_history = []

    #Our training loop
    for epoch in range(1, num_epochs + 1):
        #Training Phase
        model.train() #set the model to train mode
        epoch_loss = 0
        for inputs, targets in tqdm(train_loader): #tqdm is for the progress bar to visualize speed
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE) #move input and target to device(GPU, else CPU)
            optimizer.zero_grad() #Zero_grad reset (clear) the gradients of the model’s parameters before performing a new optimization step
            
            #forward pass
            outputs = model(inputs) #we want the output dimension to be exactly as the input
            outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False) #F.interpolate = resizing tensors, bilinear is used to resize 2D data
            #loss calculation
            loss = criterion(outputs, targets) #use MSELoss to calculate loss from outputs and targets
            #gradient calculation
            loss.backward() #computes the gradients of the loss with respect to each of the model’s parameters using backpropagation
            #update the model’s parameters
            optimizer.step() 
            #keep track of total loss
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Validation Phase
        model.eval() #set the model to evaluation mode
        val_loss = 0
        with torch.no_grad(): #temporarily disable gradient computation
            for val_inputs, val_targets in tqdm(val_loader):
                val_inputs, val_targets = val_inputs.to(DEVICE), val_targets.to(DEVICE)
                val_outputs = model(val_inputs)
                val_outputs = F.interpolate(val_outputs, size=val_targets.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(val_outputs, val_targets)
                val_loss += loss.item()
                
                # Plot prediction vs ground truth for inspection
                plot_prediction_vs_ground_truth(val_outputs[0], val_targets[0])

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

    #preparing the data to train
    file_dir = '/Users/archer/Downloads/weather_data'
    data_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith(".nrrd")]
    dataset = DataLoader(data_files=data_files)

    # Split the dataset into 80% training and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) #batch size = 1 to train it easier but slower
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=2, out_channels=2) #input file dimension is [2,1800,1420], we want the output to be the same as input
    model = model.to(DEVICE) #GPU or CPU Device, if available

    # Load the latest checkpoint if available
    checkpoint_path = '/Users/archer/Downloads/weather_data/model_weights_epoch_latest.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path) #Loads our serialized object: saved model weights
        model.load_state_dict(checkpoint['model_state_dict']) #model_state_dict access the state dictionary of our checkpoint that maps each layer’s name to its corresponding tensor of weights and biases
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("No checkpoint found, starting training from scratch.")
        start_epoch = 1

    train_model(model, train_loader, val_loader, num_epochs=10000, save_path=checkpoint_path)
