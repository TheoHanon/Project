from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import *

import numpy as np
np.random.seed(2885)
import os
import copy
from rich.progress import Progress

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # INPUTS: 
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - resultsPath (str): path to the folder containing the results of the 
    #                          experiement
    # --------------------------------------------------------------------------------
    def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        self.imgDirectory  = imgDirectory
        self.maskDirectory = maskDirectory
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        # self.model = Net(param).to(self.device)
        self.model = UNetLight(param).to(self.device)
        # -------------------
        # TRAINING PARAMETERS
        # -------------------
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain    = OxfordPetDataset(imgDirectory, maskDirectory, "train", param)
        self.dataSetVal      = OxfordPetDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest     = OxfordPetDataset(imgDirectory, maskDirectory, "test",  param)
        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
        self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
        self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)


    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl'))

    # -----------------------------------
    # TRAINING LOOP (fool implementation)
    # -----------------------------------
    def train(self):

        losses_train = []
        losses_val   = []
    
        with Progress() as progress:
            epoch_task = progress.add_task("[cyan]Epoch Progress", total=self.epoch)
            
            for epoch in range(self.epoch):
    
                batch_task = progress.add_task(f"[green]Training Epoch {epoch + 1}/{self.epoch}", total=len(self.trainDataLoader))
                
                # Training loop
                self.model.train()  # Set model to training mode
                train_loss = 0.0
                for images, GT, resizedImg in self.trainDataLoader:
                    images, GT = images.to(self.device), GT.to(self.device)

                    outputs = self.model(images).squeeze(1)
                    loss = self.criterion(outputs, GT)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                    # Update the batch progress bar
                    progress.update(batch_task, advance=1)
                    break
    
                train_loss /= len(self.trainDataLoader)
                progress.remove_task(batch_task)

                # Validation loop
                self.model.eval()  # Set model to evaluation mode
                val_loss = 0.0
   
                val_batch_task = progress.add_task(f"[blue]Validating Epoch {epoch + 1}/{self.epoch}", total=len(self.valDataLoader))
                
                with torch.no_grad():
                    for images, GT, resizedImg in self.valDataLoader:
                        images, GT = images.to(self.device), GT.to(self.device)

                        outputs = self.model(images).squeeze(1)
                        loss = self.criterion(outputs, GT)
                        val_loss += loss.item()

   
                        progress.update(val_batch_task, advance=1)
                        break

                # Calculate average validation loss 
                val_loss /= len(self.valDataLoader)
                progress.remove_task(val_batch_task)

    
                progress.update(epoch_task, advance=1)

                losses_train.append(train_loss)
                losses_val.append(val_loss)
                # Print summary for the epoch
                print(f'Epoch {epoch+1}/{self.epoch}, '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Validation Loss: {val_loss:.4f}')


        fig = plt.figure(dpi=100)
        plt.plot(losses_train, label='train', color = "blue")
        plt.plot(losses_val, label='val', color = "green")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc = "upper left", shadow = True)
        plt.savefig(self.resultsPath + '/losses.pdf')
        plt.show()

    
        # Save the model weights
        wghtsPath = self.resultsPath + '/_Weights/'
        createFolder(wghtsPath)
        torch.save(self.model.state_dict(), wghtsPath + '/wghts.pkl')



    # -------------------------------------------------
    # EVALUATION PROCEDURE (ultra basic implementation)
    # -------------------------------------------------
    def evaluate(self):
        self.model.train(False)
        self.model.eval()
        
        # Qualitative Evaluation 
        allInputs, allPreds, allGT = [], [], []
        for (images, GT, resizedImg) in self.testDataLoader:
            images      = images.to(self.device)
            predictions = (self.model(images) > 0.5).float()

            images, predictions = images.to('cpu'), predictions.to('cpu')

            allInputs.extend(resizedImg.data.numpy())
            allPreds.extend(predictions.data.numpy())
            allGT.extend(GT.data.numpy())

        allInputs = np.array(allInputs)
        allPreds  = np.array(allPreds)
        allGT     = np.array(allGT)

        showPredictions(allInputs, allPreds, allGT, self.resultsPath)

        # Quantitative Evaluation
        # Implement this ! 

