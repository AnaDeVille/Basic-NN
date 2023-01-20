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
model_path = r'C:\Users\20167271\OneDrive - TU Eindhoven\Desktop\CompVis3D\AutoEncoder\complexAutoencoder.pt'
model_path_NoisyTrained = r'C:\Users\20167271\OneDrive - TU Eindhoven\Desktop\CompVis3D\AutoEncoder\complexAutoencoderNoisy.pt'
model_path_SSIM = r'C:\Users\20167271\OneDrive - TU Eindhoven\Desktop\CompVis3D\AutoEncoder\complexAutoencoderSSIM.pt'
learning_rate = 4e-3
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
    def __init__(self, dimensions=28*28, n_codes=16):
        super(Encoder, self).__init__()
        self.n_dimensions = dimensions
        self.e_hidden1 = 392
        self.e_hidden2 = 196
        self.e_hidden3 = 98
        self.e_hidden4 = 49
        self.e_hidden5 = 32
        self.e_codes = n_codes

        self.encoder = nn.Sequential(
            nn.Flatten(-2,-1),

            nn.Linear(in_features=self.n_dimensions, out_features=self.e_hidden1),
            #nn.LeakyReLU(negative_slope=0.05, inplace=False),
            nn.BatchNorm1d(num_features=1, eps=1e-05, momentum=0.1, affine=True, device=device, dtype=dtype),
            nn.Tanh(),
            
            nn.Linear(in_features=self.e_hidden1, out_features=self.e_hidden2),
            #nn.LeakyReLU(negative_slope=0.05, inplace=False),
            nn.BatchNorm1d(num_features=1, eps=1e-05, momentum=0.1, affine=True, device=device, dtype=dtype),
            nn.Tanh(),
  
            nn.Linear(in_features=self.e_hidden2, out_features=self.e_hidden3),
            #nn.LeakyReLU(negative_slope=0.05, inplace=False),
            nn.BatchNorm1d(num_features=1, eps=1e-05, momentum=0.1, affine=True, device=device, dtype=dtype),
            nn.Tanh(),

            nn.Linear(in_features=self.e_hidden3, out_features=self.e_hidden4),
            #nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.BatchNorm1d(num_features=1, eps=1e-05, momentum=0.1, affine=True, device=device, dtype=dtype),
            nn.Tanh(),

            nn.Linear(in_features=self.e_hidden4, out_features=self.e_hidden5),
            #nn.ReLU(),
            nn.BatchNorm1d(num_features=1, eps=1e-05, momentum=0.1, affine=True, device=device, dtype=dtype),
            nn.Tanh(),

            nn.Linear(in_features=self.e_hidden5, out_features=self.e_codes),
            nn.Tanh()
        )

    def forward(self,in_):

        return self.encoder(in_), self.encoder[0](in_)#.squeeze(1))


class Decoder(nn.Module):
    """Decodes the output of the encoder into a new image"""
    def __init__(self, input_shape=16, outshape=[28,28]):
        super(Decoder, self).__init__()
        self.input_shape = input_shape
        self.d_hidden1 = 16
        self.d_hidden2 = 64
        self.d_hidden3 = 256
        self.d_flat = outshape[0] * outshape[1]
        self.outshape = outshape

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.input_shape, out_features=self.d_hidden1),  # -> 16
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),      # -> 32

            nn.Linear(in_features=self.d_hidden1 * 2, out_features=self.d_hidden2),  # -> 64
            nn.Tanh(),
            nn.Upsample(scale_factor=3, mode='nearest'),      # -> 192

            nn.Linear(in_features=self.d_hidden2 * 3, out_features=self.d_hidden3),  # -> 256
            nn.Sigmoid(),

            nn.Linear(in_features=self.d_hidden3, out_features=self.d_flat),   # -> 784
            nn.Sigmoid()
        )

    def forward(self, codes):
        assert (codes.size(-1) == self.input_shape), "The provided codes to tne decoder have the wrong length. This decoder accepts {self.input_shape} codes."
        flat = self.decoder(codes)#.unsqueeze(1))
        #out = torch.reshape(flat, [len(codes), 1]+self.outshape)

        return flat


class complexAutoEncoder(nn.Module):
    """Put the encoder and decoder together"""
    def __init__(self, img_shape=[28, 28], n_codes=16):
        super(complexAutoEncoder, self).__init__()

        # initialize decoder and encoder and their weights according to a normal distribution
        self.encoder = Encoder(dimensions=img_shape[0]*img_shape[1], n_codes=n_codes)
        self.encoder.apply(weights_init)
        self.decoder = Decoder(input_shape=n_codes, outshape=img_shape)
        self.decoder.apply(weights_init)

    def forward(self, in_):
        coded, inflat = self.encoder(in_)
        out0 = self.decoder(coded)
        out = torch.reshape(out0, [in_.size(0), 1]+[28,28])

        return out, out0, inflat





# %% Train
# initialize architecture
model = complexAutoEncoder()

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

        out, outflat, inflat = model(x_clean)
        #loss = criterion(out, x_clean)
        loss = criterion(outflat, inflat)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # learn from noisy images
    if useNoisyImgs:
        print("\n Train with noisy images as well.")
        for batch_idx,(x_clean, label) in enumerate(tqdm(loaderAugmented)):

            x_clean.to(device=device, dtype=dtype)
            label.to(device=device, dtype=dtype)
            optimizer.zero_grad()

            out, outflat, inflat = model(x_clean)
            #lossn = criterion(out, x_clean.to(torch.device(device)))
            lossn = criterion(outflat, inflat)
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

        for (data_clean, labels) in loaderVAL:
            # validation on noisy part or not
            x = data_clean
            # cast the inputs to the device
            x = x.to(device=device)
            out, outflat, inflat = model(x)
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