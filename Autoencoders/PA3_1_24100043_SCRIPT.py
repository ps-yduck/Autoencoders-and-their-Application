# %% [markdown]
# # CS 437/5317 - Deep Learning - Programming Assignment 3 - Part 1

# %% [markdown]
# ### These are all the imports you'll need for this part. If you want to add more, run it through me.

# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
from torchsummary import summary
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.notebook import tqdm
import numpy as np
import torchvision
from torch import nn, optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transform = transforms.Compose([
    transforms.ToTensor(),
])
from helper import (
    noisy_images,
    training_loop_supervised,
    training_loop_unsupervised,
    visualize_reconstructions,
    evaluate
)

BATCH_SIZE = 128 # feel free to change this & use this variable whenenver you're asked for a batch size


# %% [markdown]
# ## Task 1: Supervised Learning with CNNs

# %% [markdown]
# ### Part 1: CIFAR-10 Classification
# 
# Load the CIFAR-10 dataset from `torchvision.datasets`, train on the training set, and evaluate on the test set. You should aim to achieve a good accuracy on this task (above 70% on the test set), so feel free to experiment!

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
train_data = torchvision.datasets.CIFAR10(root='./cifar10-data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_data = torchvision.datasets.CIFAR10(root='./cifar10-data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# %% [markdown]
# ### Visualizing the Dataset

# %%
# Visualizing the data:
def imshow(img):
    img = img
    npimg = img.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.title("Random Batch of Images from the CIFAR-10 Dataset")
    plt.show()

# Get a batch of training data
dataiter = iter(train_loader)
images, _ = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images[0:32]))


# %% [markdown]
# #### Define your CNN Class below.
# Add multiple convolutional layers, as well as Batch Normalization and dropout layers along with the ReLU activation function in the convolutional block/encoder. Follow it up with a classifier layer that takes in the logits and spits out the class probabiltiies.

# %%
class CNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(32*4*4, 80),
            nn.ReLU(),

        )
        self.classifier = nn.Sequential(
            nn.Linear(80, 10),
            # nn.Dropout(dropout_rate),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Code here
        x = self.encoder(x)

        x = self.classifier(x)
        return x


# %% [markdown]
# ### Implement a training loop function for Supervised Learning in `helper.py`.
# 
# This function should return the training loss & test accuracy history, as well as save the model with the best test accuracy.

# %%
model = CNN()


model.to(device)
summary(model, input_size=(3, 32, 32), batch_size=BATCH_SIZE)

# %%


model = CNN()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20


model.to(device)
#./PA3-1_Part1_Task1_24100043.pth
model_path = "./Cifar10_CNN_classifier_PA3_rollnumber.pth"
# chekpoint = torch.load(model_path)
# model.load_state_dict(chekpoint['model_state_dict'])
# optimizer.load_state_dict(chekpoint['optimizer_state_dict'])



train_hist, test_hist = training_loop_supervised(model, train_loader, test_loader, num_epochs,
                                                 criterion, optimizer, device, model_path)


#did training on multiple different times so used load check point so graphs only show history of last run


# %%
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.6f}")

# %%




plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)

sns.set_style(style="darkgrid")
train_hist = np.array([tensor.item() for tensor in train_hist])
sns.lineplot(data=train_hist, label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")


plt.subplot(1, 2, 2)
sns.set_style(style="darkgrid")
test_hist = np.array([tensor.item() for tensor in test_hist])
sns.lineplot(data=test_hist, label="test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy")
plt.tight_layout()
plt.show()




# %% [markdown]
# ## Task 2: Image Reconstruction and Denoising

# %% [markdown]
# 
# 
# Load the MNIST dataset into memory from `torchvision.datasets`. Our first task is to create a hierarchical Linear model that sequentially reduces the spatial dimensions of the images, encodes/compresses them from `28x28` images to just `10` features in a compact latent space representation, and then reconstruct the images using those 10 features. This type of an Encoder-Decoder architecture can be efficiently used for image compression (although there's a computational overhead)

# %%

transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = torchvision.datasets.MNIST(root='./mnist-data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_data = torchvision.datasets.MNIST(root='./mnist-data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)



# %%

# Create your Linear AutoEncoder Class:

class LinearAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 15)
        )
        self.decoder = nn.Sequential(

            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()

        )
    def forward(self, x):


        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)

        return x




# %% [markdown]
# #### Define a function that adds Gaussian noise to images (remember to clamp the values b/w 0 and 1) in `helper.py`

# %% [markdown]
# ### Write the training loop for unsupervised training in `helper.py`
# 
# **Note**: It'll be very similar to the supervised training loop, so feel free to copy that and make the necessary changes
# 
# For denoising, add noise to your image before passing it to the model. You must use the default parameters of the `noisy_images` function

# %%

model = LinearAutoEncoder()

model.to(device)
summary(model, input_size=(1, 28, 28))

# %% [markdown]
# ### Train your model. Use MSELoss & AdamW optimizer

# %%
# Code here
model = LinearAutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
model.to(device)
#./PA3-1_Part1_Task2_Linear_AE_recons_24100043.pth
model_path = "./MNIST10_LinearAutoEncoder_PA3_rollnumber.pth" # change this according to where you wanna save the model weights
num_epochs = 20
train_hist, test_hist = training_loop_unsupervised(model, "reconstruction",train_loader, test_loader, num_epochs,
                                                 criterion, optimizer, device, model_path)



# %% [markdown]
# ### Plot the test loss history

# %%




plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)

sns.set_style(style="darkgrid")
train_hist = np.array([tensor.item() for tensor in train_hist])
sns.lineplot(data=train_hist, label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")


plt.subplot(1, 2, 2)
sns.set_style(style="darkgrid")
test_hist = np.array([tensor.item() for tensor in test_hist])
sns.lineplot(data=test_hist, label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Test Loss")
plt.tight_layout()
plt.show()




# %% [markdown]
# ### Visualizing the model's reconstructions:

# %%


visualize_reconstructions(test_loader, model, device, 'reconstruction')

# %% [markdown]
# ### Now do denoising using a new instance of the LinearAutoEncoder class

# %%
# Code here

model = LinearAutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

model.to(device)
#./PA3-1_Part1_Task2_Linear_AE_denoise_24100043.pth
model_path = "./LinearAutoEncoder_denoise_PA3_rollnumber.pth" # change this according to where you wanna save the model weights
num_epochs = 25
train_hist, test_hist = training_loop_unsupervised(model, "denoising",train_loader, test_loader, num_epochs,
                                                 criterion, optimizer, device, model_path)

# %% [markdown]
# #### Plot the test loss history

# %%




plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)

sns.set_style(style="darkgrid")
train_hist = np.array([tensor.item() for tensor in train_hist])
sns.lineplot(data=train_hist, label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")


plt.subplot(1, 2, 2)
sns.set_style(style="darkgrid")
test_hist = np.array([tensor.item() for tensor in test_hist])
sns.lineplot(data=test_hist, label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Test Loss")
plt.tight_layout()
plt.show()




# %% [markdown]
# ### Visualizing the Denoising

# %%
visualize_reconstructions(test_loader, model, device, 'denoising')

# %% [markdown]
# #### A massive drawback of a simple linear architecture is that we lose a ton of spatial information of the image when we flatten it. Moreover, treating each pixel as one input has a massive computational overhead, especially when you're dealing with higher resolution images (e.g., in ImageNET, the 3x224x224 images would lead to 150,528 inputs!). You might even see some poor denoising by this model. Generally, for image data, convolutional architectures are far more popular, and you'll see why in this next part.
# 
# Using the Convolutional Encoder-Decoder Model described in the instructions, do both image reconstruction and denoising

# %%
class AutoEncoder_CNN(nn.Module):
    # Code here
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), #14x14
            nn.MaxPool2d(kernel_size=2, stride=2), #7x7
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(), #4x4
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            #2x2
        )
    
        '''
        Hout =(H_in-1)xstride[0]-2xpadding[0]+dilation[0]x(kernel_size[0]-1)+output_padding[0]+1
        '''
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=7, stride=2, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x , indices = self.encoder(x)
        x = self.unpool(x, indices)

        x = self.decoder(x)

        return x

# %%
model = AutoEncoder_CNN()
summary(model.to(device), input_size=(1, 28, 28))

# %% [markdown]
# Train your model:

# %%
# Code here
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
model.to(device)
#./PA3-1_Part1_Task2_CNN_AE_recons_24100043.pth
model_path = "./MNIST_AutoEncoder_reconstruction_CNN_PA3_rollnumber.pth" # change this according to where you wanna save the model weights
num_epochs = 20
train_hist, test_hist = training_loop_unsupervised(model, "reconstruction",train_loader, test_loader, num_epochs,
                                                 criterion, optimizer, device, model_path)


# %%




plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)

sns.set_style(style="darkgrid")
train_hist = np.array([tensor.item() for tensor in train_hist])
sns.lineplot(data=train_hist, label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")


plt.subplot(1, 2, 2)
sns.set_style(style="darkgrid")
test_hist = np.array([tensor.item() for tensor in test_hist])
sns.lineplot(data=test_hist, label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Test Loss")
plt.tight_layout()
plt.show()




# %% [markdown]
# ## Visualization

# %%
visualize_reconstructions(test_loader, model, device, 'reconstruction')

# %% [markdown]
# ### Now do the same for Denoising:

# %%
# Code here
model = AutoEncoder_CNN()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
model.to(device)
#./PA3-1_Part1_Task2_CNN_AE_denoise_24100043.pth
model_path = "./MNIST_AutoEncoder_CNN_denoise_PA3_rollnumber.pth" # change this according to where you wanna save the model weights
num_epochs = 15
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

train_hist, test_hist = training_loop_unsupervised(model, "denoising",train_loader, test_loader, num_epochs,
                                                 criterion, optimizer, device, model_path)



# %% [markdown]
# ### Plot the test_loss history

# %%




plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)

sns.set_style(style="darkgrid")
train_hist = np.array([tensor.item() for tensor in train_hist])
sns.lineplot(data=train_hist, label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")


plt.subplot(1, 2, 2)
sns.set_style(style="darkgrid")
test_hist = np.array([tensor.item() for tensor in test_hist])
sns.lineplot(data=test_hist, label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Test Loss")
plt.tight_layout()
plt.show()




# %% [markdown]
# ### Visualizing the Denoising of the Convolutional AE model

# %%
visualize_reconstructions(test_loader, model, device, 'denoising')

# %% [markdown]
# Clearly far better, right? And with much less parameters!

# %% [markdown]
# #### Compare the number of parameters of both models, and comment on how/why a convolutional neural network performs so much better on image data compared to a fully connected network while using far less parameters.
# 
# - Answer here:

# %% [markdown]
# 

# %%



