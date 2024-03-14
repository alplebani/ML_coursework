#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Helpers.HelperFunctions import ddpm_schedules, CNNBlock, CNN, DDPM,PersonalDegradation


import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid


plt.style.use('mphil.mplstyle')


def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plots', help='Flag: if selected, will show the plots instead of only saving them', required=False, action='store_true')
    parser.add_argument('-n', '--nepochs', help='Number of epochs you want to run on', required=False, default=100, type=int)
    parser.add_argument('--easy', help='Run a simplified version of the network for testing', action='store_true')
    parser.add_argument('-l', '--layers', nargs='+', type=int, default=(8,8), required=False)
    parser.add_argument('-e-','--eta', help='Learning rate', type=float, default=2e-4, required=False)
    parser.add_argument('-b', '--beta', help='Noise schedule beta', type=float, nargs='+', default=(1e-4, 0.02), required=False)
    parser.add_argument('--nT', help='Number of diffusion steps', type=int, default=1000, required=False)
    parser.add_argument('--name', help='Name of the model', type=str, required=True)
    parser.add_argument('-t', '--type', help='Type of model you want to use for degradation', choices=['DDPM', 'Personal'], required=False, default='DDPM', type=str)
    parser.add_argument('-d', '--drop', help='Dropout rate for pixels in the image, to be used only with the Personal model', required=False, type=float)
    parser.add_argument('-r', '--range', help='Range in which  the pixel brightness is adjusted', required=False, type=float, nargs='+')
    args = parser.parse_args()
    
    torch.manual_seed(4999) # set a random seed as my birthday to have reproducible code 
    torch.device('cpu') # set by default the CPU, as I'm running on CPU on my laptop   
    
    # creating the folders where to store plots and models if they don't exist
    
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    if not os.path.exists('contents'):
        os.makedirs('contents')
    if not os.path.exists('model'):
        os.makedirs('model')   
    
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    
    # Setting of hyperparameters, passed by command line
    
    
    layers = args.layers        
    LR = args.eta
    beta = args.beta
    nT = args.nT
    name = args.name
    my_type = args.type
    
    if my_type == 'Personal' and not (args.drop and args.range):
        print('Error! If running with the personal model you need to provide a dropout and a range')
        print('Exiting code')
        print('==================================')
        exit(0)
    
    if my_type == 'Personal':
        dropout = args.drop
        my_range = args.range
    
    print('==================================')
    print('Summary of hyperparameters:')
    print('----------------------------------')
    
    print(f'Layers = {layers}')
    print(f'Learning Rate = {LR}')
    print(f'Beta ={beta}')
    print(f'nT = {nT}')
    print(f'Name of the model = {name}')
    print(f'Type = {my_type}')
    
    print('==================================')
    
    if not args.layers:
        if args.easy:
            gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(8,8), act=nn.GELU)
        else:
            gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(32,64,64,32), act=nn.GELU)
    
    else:
        gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=layers, act=nn.GELU)
    # For testing: (16, 32, 32, 16)
    # For more capacity (for example): (64, 128, 256, 128, 64)
    ddpm = DDPM(gt=gt, betas=beta, n_T=nT)
    optim = torch.optim.Adam(ddpm.parameters(), lr=LR)
    
    accelerator = Accelerator()

    # We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
    # which lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
    ddpm, optim, dataloader = accelerator.prepare(ddpm, optim, dataloader)
    
    n_epoch = args.nepochs
    losses = []
    best_loss = float('inf') # start with -infinity so that the first step represents always an improvement in the loss
    delta_ES = 0.005 # delta for early stopping
    patience = 5 # number of epochs to use to evaluate the early stopping
    epochs_run_on = []
    best_losses = []
    patience_best_losses = []
    
    # breakpoint()

    for i in range(n_epoch):
        epochs_run_on.append(i) # save only the epochs you actually run on for the loss function plot
        ddpm.train()

        pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar
        for x, _ in pbar:
            optim.zero_grad()
            
            if my_type == 'Personal':
                degradation = PersonalDegradation(dropout=dropout, my_range=my_range)
                x = degradation(x)

            loss = ddpm(x)

            loss.backward()

            losses.append(loss.item())
            avg_loss = np.average(losses[min(len(losses)-100, 0):])
            pbar.set_description(f"Epoch : {i}, loss: {avg_loss:.3g}")  # Show running average of loss in progress bar

            optim.step()
        
        early_stop = False
        patience_best_losses.append(avg_loss)
        if i > patience:
            if np.abs(best_loss - patience_best_losses[-(patience + 1)]) <= delta_ES:
                early_stop = True
                
            
        if early_stop:
            print(f"Early stopping at epoch {i} due to minimal loss improvement")
            best_losses.append(best_loss)
            # breakpoint()
            break
        else:
            best_loss = min(best_loss, avg_loss) 
            best_losses.append(best_loss)
        
        ddpm.eval()
        
        
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), accelerator.device) 
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"contents/{name}_{my_type}_sample_{i:04d}.png")

            torch.save(ddpm.state_dict(), f"model/{name}_{my_type}_mnist.pth")

    image, _ = next(iter(dataloader)) 

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 5))  # Adjust as needed
    for i, ax in enumerate(axes.flat):
        ax.imshow(image[i].squeeze(0), cmap='gray')
        ax.axis('off')
    
    
    plt.savefig('Plots/example_MNIST.pdf')
    print('Example MNIST picture saved at Plots/example_MNIST.pdf')
    print('===================================')
    # Plotting loss function
    plt.figure()
    plt.plot(epochs_run_on, best_losses)
    plt.title(f'Loss function for model {my_type}')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.savefig(f'Plots/{name}_{my_type}_loss_function.pdf')
    print('===================================')
    print(f'Plot saved at Plots/{name}_{my_type}_loss_function.pdf')
    print('===================================')
       
    if args.plots: # display plots only if the --plots is used
        plt.show()


if __name__ == "__main__":
    print("=======================================")
    print("Initialising coursework")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Coursework finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")