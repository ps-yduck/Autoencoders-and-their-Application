import torch, os, cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
])


def noisy_images(images, mean=0.0, std=0.5):
    """
    Adds Gaussian noise to a batch of images.

    Parameters:
    - images: tensor of images of shape (N, C, H, W), where N is the batch size,
              C is the number of channels, H is the height, and W is the width.
    - mean: Mean of the Gaussian noise.
    - std: Standard deviation of the Gaussian noise.

    Returns:
    - Tensor of noisy images of the same shape as the input.
    """
    noisy_images = images + (torch.randn(images.size()).to(images.device) * std + mean)
    noisy_images = torch.clamp(noisy_images, 0, 1) # limits to 0 and 1
    return noisy_images
    

def evaluate (model, dataloader, criterion, device):
    
    model.eval()
    tp_tn = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss = criterion(pred, labels)
            _, preds = torch.max(pred, 1)
            loss += loss.item() * images.size(0)
            tp_tn += torch.sum(preds == labels.data)
            total += images.size(0)
    validation_loss = loss / total
    validation_accuracy = tp_tn / total
    return validation_loss, validation_accuracy

def unsuperivsed_evaluate (model, dataloader, criterion, device, task):
    model.eval()
    loss = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images = None
            labels = None
            if len(batch) == 2:
                images, labels = batch
            else:
                images = batch
            images = images.to(device)
            if task == 'denoising':
                images = noisy_images(images)
            pred = model(images)
            loss = criterion(pred, images)
            loss += loss.item() * images.size(0)
            total += images.size(0)
    validation_loss = loss / total
    return validation_loss

def training_loop_supervised(model, train_loader, test_loader, num_epochs,
                             criterion, optimizer, device, model_path):
    """
    Trains and evaluates a PyTorch image classification model, saving the model with the best accuracy on the test dataset.

    Args:
    - model: The PyTorch model to be trained and evaluated.
    - train_loader: DataLoader for the training data.
    - test_loader: DataLoader for the testing data.
    - num_epochs: Number of epochs to train the model.
    - criterion: Loss function used for training.
    - optimizer: Optimization algorithm used for training.
    - device: The device ('cuda' or 'cpu') to perform training and evaluation on.
    - model_path: Path to save the model achieving the best accuracy on the test set.

    Returns:
    - train_loss_history: List of average training losses for each epoch.
    - test_accuracy_history: List of test accuracies for each epoch.
    """
   
    model.to(device)
    train_loss_history = []
    test_accuracy_history = []
    best_test_accuracy = 0.0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
   
    for epoch in tqdm(range(num_epochs)):
    
        model.train()
        loss = 0
        tp_tn = 0
        total = 0

        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # print(images.shape)
            # print(labels.shape)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels) #avg loss per batch
            loss.backward()
            optimizer.step()
            # print (loss)
            loss += loss.item() * labels.size(0) # same as batch size. this is weigh
            _, predicted = torch.max(pred, 1)
            tp_tn += (predicted == labels).sum().item()
            total += labels.size(0) # same as batch size
        
        # print (len(train_dataloader.dataset))
        # this is total images in train loader. same as variable total
        train_loss = loss / total
        train_accuracy = tp_tn / total

        
        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}")

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print (f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}")

        scheduler.step(test_accuracy)

        train_loss_history.append(train_loss)
        test_accuracy_history.append(test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            # best_model = model.state_dict()
            # torch.save(best_model, model_path)
            save_checkpoint(model, optimizer, epoch+1, model_path)
        

    
    
    return train_loss_history, test_accuracy_history

# Training Loop function for unsupervised learning (can be used for both reconstruction and denoising)

def training_loop_unsupervised(model, task, train_loader, test_loader, num_epochs,
                             criterion, optimizer, device, model_path):
    """
    Trains and evaluates a PyTorch image classification model, saving the model with the best accuracy on the test dataset.

    Args:
    - model: The PyTorch model to be trained and evaluated.
    - task: string, either 'reconstruction' or 'denoising'
    - train_loader: DataLoader for the training data.
    - test_loader: DataLoader for the testing data.
    - num_epochs: Number of epochs to train the model.
    - criterion: Loss function used for training.
    - optimizer: Optimization algorithm used for training.
    - device: The device ('cuda' or 'cpu') to perform training and evaluation on.
    - model_path: Path to save the model achieving the best accuracy on the test set.

    Returns:
    - train_loss_history: List of average training losses for each epoch.
    - test_accuracy_history: List of test losses for each epoch.
    """
    model.to(device)
    train_loss_history = []
    test_accuracy_history = []
    minimum_test_loss = float('inf')
   
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss = 0
       
        total = 0

        for batch in train_loader:
           
            images = None
            labels = None
            if len(batch) == 2:
                images, labels = batch
            else:
                images = batch


            images = images.to(device)
            if task == 'denoising':
                images = noisy_images(images)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, images)
            loss.backward()
            optimizer.step()
            loss += loss.item() * images.size(0)
            
            total += images.size(0)
        
        train_loss = loss / total

        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")

        test_loss = unsuperivsed_evaluate(model, test_loader, criterion, device, task)
        print (f"Test Loss: {test_loss:.6f}")
        
        scheduler.step(test_loss)
        train_loss_history.append(train_loss)
        test_accuracy_history.append(test_loss) #test loss actually

        if test_loss < minimum_test_loss:
            minimum_test_loss = test_loss
            # best_model = model.state_dict()
            # torch.save(best_model, model_path)
            save_checkpoint(model, optimizer, epoch+1, model_path)



    # Code here


    return train_loss_history, test_accuracy_history

def segmentation_evaluate(model, dataloader, criterion, device, epoch):
    model.eval()
    loss = 0
    total = 0
    sample_outputs = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            pred = model(images)
            loss = criterion(pred, masks)
            loss += loss.item() * images.size(0)
            total += images.size(0)
            if  epoch % 5 == 0 and len(sample_outputs) < 2:  # Store 2 sample outputs every 5 epochs picked from first two batches
                random_index = np.random.choice(images.size(0))
                sample_outputs.append({
                    'image': images[random_index].cpu(),
                    'predicted_mask': pred[random_index].cpu(),
                    'ground_truth_mask': masks[random_index].cpu() 
                })
    validation_loss = loss / total
    
    return validation_loss, sample_outputs


def training_loop_segmentation(model, train_loader, val_loader, num_epochs,
                               criterion, optimizer, device, model_path):
    """
    Trains and evaluates a PyTorch segmentation model, saving the model with the best performance on the validation dataset.
    Stores 10 sample outputs every 10 epochs.

    Args:
    - model: The PyTorch model to be trained and evaluated.
    - train_loader: DataLoader for the training data.
    - val_loader: DataLoader for the validation data.
    - num_epochs: Number of epochs to train the model.
    - criterion: Loss function used for training.
    - optimizer: Optimization algorithm used for training.
    - device: The device ('cuda' or 'cpu') to perform training and evaluation on.
    - model_path: Path to save the model achieving the best performance on the validation set.

    Returns:
    - train_loss_history: List of average training losses for each epoch.
    - val_loss_history: List of validation losses for each epoch.
    - sample_outputs: List of dictionaries containing images, predicted masks, and ground truth masks for sample outputs.
    """

   
    

    model.to(device)
    train_loss_history = []
    val_loss_history = []
    sample_outputs = []
    minimum_test_loss = float('inf')
 

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss = 0
       
        total = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, masks)
            loss.backward()
            optimizer.step()
            loss += loss.item() * images.size(0)
            
            total += images.size(0)
        
        train_loss = loss / total

        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")

        val_loss, two_sample_out = segmentation_evaluate(model, val_loader, criterion, device, epoch+1)
        print (f"Validation Loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        if two_sample_out != []:
            sample_outputs.append(two_sample_out)

        if val_loss < minimum_test_loss:
            minimum_test_loss = val_loss
           
            # torch.save(best_model, model_path)
            save_checkpoint(model, optimizer, epoch+1, model_path)

    # Code here
    
    
    return train_loss_history, val_loss_history, sample_outputs


def visualize_reconstructions(test_loader, model, device, task='reconstruction'):
    """
    Visualizes original and reconstructed images from a test dataset.

    Args:
    - test_loader: DataLoader for the test dataset.
    - model: Trained model for image reconstruction or denoising.
    - device: The device (e.g., 'cuda' or 'cpu') the model is running on.
    - task: The task to perform - either 'reconstruction' or 'denoising'.
    """
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    if task == 'denoising':
        images = noisy_images(images)  # Apply noise if the task is denoising

    recons = model(images.to(device))

    images_np = images.cpu().numpy()
    recons_np = recons.cpu().detach().numpy()

    indices = np.random.choice(images_np.shape[0], 9, replace=False)
    selected_images = images_np[indices]
    selected_recons = recons_np[indices]
    selected_labels = labels.numpy()[indices]

    fig, axs = plt.subplots(2, 9, figsize=(15, 4))

    for i in range(9):
        axs[0, i].imshow(np.transpose(selected_images[i], (1, 2, 0)), interpolation='none', cmap='gray')
        axs[0, i].set_title(f"GT: {selected_labels[i]}")
        axs[0, i].axis('off')
        axs[1, i].imshow(np.transpose(selected_recons[i], (1, 2, 0)), interpolation='none', cmap='gray')
        axs[1, i].set_title(f"Recon: {selected_labels[i]}")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()




# Dataset class for Loading Unlabeled Images:
class UnlabeledImageDataset(Dataset):
    def __init__(self, folder, split='train', split_ratio=0.8):
        self.folder = folder
        self.imgs = os.listdir(folder)
        self.split = split
        self.split_ratio = split_ratio
        
        num_imgs = len(self.imgs)
        self.split_index = int(num_imgs * split_ratio)
        
        if split == 'train':
            self.imgs = self.imgs[:self.split_index]
        elif split == 'val':
            self.imgs = self.imgs[self.split_index:]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.imgs[idx])
        img_np = cv2.imread(img_path)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_pt = transform(img_np) # converting to pytorch tensor
        return img_pt
 
class UnlabeledImageDataset_2(Dataset):
    def __init__(self, folder, split='train', split_ratio=0.8):
        self.folder = folder
        self.transform = transform
        all_imgs = os.listdir(folder)
        num_imgs = len(all_imgs)
        split_index = int(num_imgs * split_ratio)

        if split == 'train':
            self.imgs = all_imgs[:split_index]
        elif split == 'val':
            self.imgs = all_imgs[split_index:]

        self.images = []
        for img_name in tqdm(self.imgs):
            img_path = os.path.join(self.folder, img_name)
            img_np = cv2.imread(img_path)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                img_np = self.transform(img_np)
                
            self.images.append(img_np)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # since images are already preloaded, just retrieve from the list
        return self.images[idx]
    
def unlabeled_dataset(ram_above_16: bool = False):
    if ram_above_16:
        return UnlabeledImageDataset_2
    else:
        return UnlabeledImageDataset


def save_checkpoint(model, optimizer, epoch, model_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)


