from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import *
import albumentations as A

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
        self.patience      = param["TRAINING"]["PATIENCE"]

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
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.3)

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
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', weights_only = True))

    # -----------------------------------
    # TRAINING LOOP (fool implementation)
    # -----------------------------------
    def train(self):
        
        best_val_loss = np.inf
        epochs_no_improve = 0

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

                # Calculate average validation loss 
                val_loss /= len(self.valDataLoader)
                self.scheduler.step()

                progress.remove_task(val_batch_task)
                progress.update(epoch_task, advance=1)

                losses_train.append(train_loss)
                losses_val.append(val_loss)
                # Print summary for the epoch
                print(f'Epoch {epoch+1}/{self.epoch}, '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Validation Loss: {val_loss:.4f},'
                    f"lr: {self.scheduler.get_last_lr()}")
                

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0  # Reset counter if we see improvement
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                

        np.savez(self.resultsPath + '/learning_curve.npz', losses_train=losses_train, losses_val=losses_val)
    
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
        
            predictions = torch.sigmoid(self.model(images))            
            images, predictions = images.to('cpu'), predictions.to('cpu')

            allInputs.extend(resizedImg.data.numpy())
            allPreds.extend(predictions.data.numpy())
            allGT.extend(GT.data.numpy())

        allInputs = np.array(allInputs)
        allPreds  = np.array(allPreds)
        allGT     = np.array(allGT)

        showPredictions(allInputs, allPreds, allGT, self.resultsPath)

        # Quantitative Evaluation
        thresholds = np.linspace(0,1,19)
        TP = np.zeros_like(thresholds)
        FP = np.zeros_like(thresholds)
        FN = np.zeros_like(thresholds)
        TN = np.empty_like(thresholds)
        Tot = 0
        NumPos = 0
        for (images, GT, resizedImg) in self.testDataLoader:
            images      = images.to(self.device)
        
            predictions = self.model(images)
            # print(predictions.shape)
            images, predictions = images.to('cpu'), predictions.to('cpu')

            predictions = np.squeeze(predictions.data.numpy())
            predictions = 1 / (1 + np.exp(-predictions))
            GT = GT.data.numpy()
            Tot += GT.size
            NumPos += np.sum(GT)
            
            for (i,t) in enumerate(thresholds):
                mask = predictions > t
                TP[i] += np.sum((mask == 1) & (GT == 1))
                FP[i] += np.sum(mask > GT)
                FN[i] += np.sum(mask < GT)
        
        for i in range(len(thresholds)):
            TN[i] = Tot - TP[i] - FP[i] - FN[i]
        NumNeg = Tot - NumPos

        TPR = TP/NumPos
        FPR = FP/NumNeg
        Precision = TP/(TP+FP)
        Recall = TPR
        Accuracy = (TP+TN)/Tot
        F1 = 2*TP/(2*TP+FP+FN)
        IoU = TP/(TP+FP+FN)
        Precision[-1] = 1
        np.savez(self.resultsPath + '/Metrics.npz', TPR=TPR, FPR=FPR, Precision=Precision, Recall=Recall, Accuracy=Accuracy, F1=F1, IoU=IoU)
