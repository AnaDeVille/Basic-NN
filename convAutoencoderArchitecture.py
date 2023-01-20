import torch
from torch import Tensor
import torch.nn.functional as F  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import sampler
from torchvision.datasets import MNIST
from torchvision import transforms 
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as T
from tqdm import tqdm
import math


########### Parameters
useNoisyImgs = True
useSSIM = False
model_path = r'C:\Users\20167271\OneDrive - TU Eindhoven\Desktop\CompVis3D\AutoEncoder\convAutoencoder.pt'
model_path_NoisyTrained = r'C:\Users\20167271\OneDrive - TU Eindhoven\Desktop\CompVis3D\AutoEncoder\convAutoencoderNoisy.pt'
learning_rate = 2e-3
no_epochs = 30




# %% Get data 

# transform dataset from PIL images to tensors containing floating point pixels
dtype = torch.float32
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(dtype)
])

# load data into DataLoader class
dir_ = r"C:\Users\20167271\OneDrive - TU Eindhoven\Desktop\CompVis3D\MLP\datasets"  # same as before
# training data
mnistTRAIN = MNIST(dir_, train=True, download=True, transform=transform)
loaderTRAIN = DataLoader(mnistTRAIN, batch_size=64, sampler=sampler.SubsetRandomSampler(range(59000)))
# subset of images to add noise
mnistSubset = MNIST(dir_, train=True, download=True, transform=transform)
loaderSubset = DataLoader(mnistTRAIN, batch_size=64, sampler=sampler.SubsetRandomSampler(range(6000)))
# validation dataset
mnistVAL = MNIST(dir_, train=True, download=True,transform=transform)
loaderVAL = DataLoader(mnistVAL, batch_size=64, sampler=sampler.SubsetRandomSampler(range(59000, 60000)))
# final testing data
mnistTEST = MNIST(dir_, train=False, download=True, transform=transform)
loaderTEST = DataLoader(mnistTEST, batch_size=64)


class augmentedDataSet(Dataset):
    """"Add noise on images of a DataLoader and create a dataset which can later be used to create a new DataLoader"""
    def __init__(self, loader):
        self.base_loader = loader
        self.possibleAugmentations = ['brightness','rotation','blur', 'erase', 'contrast']
        self.x_data = None
        self.y_data = None

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        if not(self.x_data == None or self.y_data==None):
            return self.x_data[idx, :, :, :], self.y_data[idx] 
        else: 
            return None

    def augmentations(self, *args):
        imgsOut = []
        labelsOut = []
        for batchImgs, batchLabels in self.base_loader:
            for t in range(batchImgs.shape[0]):
                x = batchImgs[t]
                y = batchLabels[t]
                y = torch.unsqueeze(y, 0)
                for _ in args:
                    if _ not in self.possibleAugmentations: 
                        print('\n Augmentation {} is not part of possible augmentations: {}.'.format(_, self.possibleAugmentations))
                        return None
                    
                    if 'bright' in _:
                        for factor in np.arange(0.5, 1.5, 0.3):
                            ax = T.adjust_brightness(x, factor)
                            ax = torch.unsqueeze(ax, 0)
                            imgsOut.append(ax)
                            labelsOut.append(y)

                    if 'contrast' in _:
                        for factor in np.arange(0.5, 1.5, 0.3):
                            ax = T.adjust_contrast(x, factor)
                            ax = torch.unsqueeze(ax, 0)
                            imgsOut.append(ax)
                            labelsOut.append(y)

                    if 'blur' in _:
                        #add gaussian blur
                        ax = T.gaussian_blur(x, kernel_size=3, sigma=50)
                        ax = torch.unsqueeze(ax, 0)
                        imgsOut.append(ax)
                        labelsOut.append(y)
                        #add salt and pepper noise
                        for factor in np.arange(0.2,0.81,0.2):
                            axsp = ax + torch.rand(ax.size(), requires_grad=False)*factor
                            axsp = torch.clip(ax, min=0.0, max=255.0)
                            imgsOut.append(axsp)
                            labelsOut.append(y)

                    if 'rotat' in _:
                        for factor in range(-20, 20, 10):
                            ax = T.rotate(x, factor)
                            ax = torch.unsqueeze(ax, 0)
                            imgsOut.append(ax)
                            labelsOut.append(y)

                    if 'erase' in _: #occlusion
                        for i in range(8, 23, 4):
                            for j in range(8, 23, 4):
                                ax = T.erase(x,i,j,4,4, x[0,0,i,j])
                                ax = torch.unsqueeze(ax, 0)
                                imgsOut.append(ax)
                                labelsOut.append(y)

        self.x_data = torch.cat(imgsOut,0)
        self.y_data = torch.cat(labelsOut,0)


newDataset = augmentedDataSet(loaderSubset)
newDataset.augmentations('blur')#('brightness','rotation','blur', 'erase', 'contrast')
loaderAugmented = DataLoader(newDataset, batch_size=64)
        


# %% Set device
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('\n using device:', device)




#%% Architecture

#for weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        torch.nn.init.xavier_uniform_(m.weight)  # I had some issues with backpropagation, it did not change much

class Encoder(nn.Module):
    """Encoder network, outputs a couple of codes to encode the image"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.input_shape = [28,28]
        self.outshape = [5,5]

        self.encoder = nn.Sequential(
            nn.Conv2d(1,4, kernel_size=(3,3),padding=1, stride=1), #[Nx1x32x32]=> [Nx4x28x28]
            nn.Tanh(),
            nn.BatchNorm2d(num_features=4),
            nn.MaxPool2d(2,2),#[Nx16x32x32]=> [Nx16x16x16]

            nn.Conv2d(4,8, kernel_size=(3,3),padding=1, stride=1), #[Nx16x16x16]=> [Nx16x14x14]
            nn.Tanh(),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(2,2),#[Nx16x16x16]=> [Nx16x8x8]

            nn.Conv2d(8,1, kernel_size=(3,3), padding=1, stride=1), #[Nx16x8x8]=> [Nx32x7x7]
            nn.Tanh(),
            nn.BatchNorm2d(num_features=1),
            nn.MaxPool2d(3,1),#[Nx32x7x7]=> [Nx16x5x5]
        )

    def forward(self,in_):

        return self.encoder(in_)


class Decoder(nn.Module):
    """Decodes the output of the encoder into a new image"""
    def __init__(self):
        super(Decoder, self).__init__()
        self.input_shape = [5,5]
        self.outshape = [28,28]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 4, kernel_size=(3,3)), # [Nx1x5x5]=>[Nx1x7x7]
            nn.Tanh(),
            nn.ConvTranspose2d(4, 8, kernel_size=(5,5)), # [Nx1x7x7]=>[Nx1x11x11]
            nn.Tanh(),

            nn.Upsample(scale_factor=2), #[Nx16x4x3]=>[Nx1x22x22]

            nn.ConvTranspose2d(8, 8, kernel_size=(3,3)), # [Nx1x22x22]=>[Nx1x24x24]
            nn.Tanh(),
            nn.ConvTranspose2d(8, 4, kernel_size=(3,3)), # [Nx1x24x24]=>[Nx1x26x26]
            nn.Tanh(),
            nn.ConvTranspose2d(4, 1, kernel_size=(3,3)), # [Nx1x26x26]=>[Nx1x28x28]
            nn.Sigmoid(),

        )

    def forward(self, codes):

        return self.decoder(codes)


class convAutoEncoder(nn.Module):
    """Put the encoder and decoder together"""
    def __init__(self):
        super(convAutoEncoder, self).__init__()

        # initialize decoder and encoder and their weights according to a normal distribution
        self.encoder = Encoder()
        self.encoder.apply(weights_init)
        self.decoder = Decoder()
        self.decoder.apply(weights_init)

    def forward(self, in_):
        coded = self.encoder(in_)
        out = self.decoder(coded)

        return out



# %% Train
# initialize architecture
model = convAutoEncoder()

# pick which loss function to use
optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, eps=1e-08, weight_decay=0.0005)
criterion = nn.MSELoss(reduction='mean')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay

eval_dic = {'Loss_t': [], 'train_acc': [],'Loss_v': [], 'valid_acc': []}
loss_train = []
loss_val = []


# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")

    # prepare for training
    running_loss = 0.0
    running_loss_noisy = 0.0
    model.train()
    # go over all minibatches
    # learn from training data
    for batch_idx,(x_clean, label) in enumerate(tqdm(loaderTRAIN)):

        x_clean.to(device=device, dtype=dtype)
        label.to(device=device, dtype=dtype)
        optimizer.zero_grad()

        out = model(x_clean)
        #loss = criterion(out, x_clean)
        loss = criterion(out, x_clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # learn from noisy images
    if useNoisyImgs:
        print("\n Train with noisy images as well.")
        for batch_idx,(x_noisy, label) in enumerate(tqdm(loaderAugmented)):

            x_noisy = x_noisy.to(device=device, dtype=dtype)
            optimizer.zero_grad()
            out = model(x_noisy)
            #lossn = criterion(out, x_clean.to(torch.device(device)))
            lossn = criterion(out, x_noisy)
            lossn.backward()
            optimizer.step()

            running_loss_noisy += lossn.item()

    # Update loss in dictionary
    train_epoch_loss = running_loss/len(loaderTRAIN)
    trainNoisy_epoch_loss = running_loss_noisy/len(loaderAugmented)
    eval_dic['Loss_t'].append(train_epoch_loss/2 + trainNoisy_epoch_loss/2)

    # Check validation loss
    with torch.no_grad():
        running_loss_val = 0.0
#         Set the model to evaluation mode
        model.eval()

        for (x, labels) in loaderVAL:
            # validation on noisy part or not
            x = x.to(device=device)
            out = model(x)
            lossv = criterion(out, x)
            running_loss_val += lossv.item()

        val_epoch_loss = running_loss_val/len(loaderVAL)
        eval_dic['Loss_v'].append(val_epoch_loss)

    scheduler.step()  # update learning rate every 10 epochs
    print('Epoch', epoch)
    print('Training Loss', eval_dic['Loss_t'][epoch])
    print('Validation Loss', eval_dic['Loss_v'][epoch])


# %% Save and check the model
def save_checkpoint(epochs:int, model:nn.Module, optimizer, loss, path:str):
    """Write function to save model checkpoint"""
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)



if useNoisyImgs:
    save_checkpoint(no_epochs, model, optimizer, criterion, model_path_NoisyTrained)
else:
    save_checkpoint(no_epochs, model, optimizer, criterion, model_path)