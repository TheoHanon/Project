# Plot 10 augmented images
import matplotlib.pyplot as plt


from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import *
import albumentations as A
import cv2
from scipy.stats import entropy
from albumentations.pytorch import ToTensorV2

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


def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


class Network_Class: 

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


        self.transforms = A.ReplayCompose([
            A.Rotate(limit=(-35, 35), p=0.7),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7), 
            A.ToGray(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Blur(blur_limit=7, p=0.3),
        ], additional_targets={'mask': 'mask'})

    
    def loadWeights(self): 
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', weights_only = True))


    def inverse_transform(self, transformed_mask, replay_data):
        transformed_mask = transformed_mask.squeeze(0).numpy()
        transformed_mask = np.transpose(transformed_mask, (1, 2, 0))
        
        for t in reversed(replay_data['transforms']):
        
            transform_name = t['__class_fullname__']
        
            if 'Rotation' in transform_name and t["applied"]:
                affine_matrix = t["params"]["matrix"]
                if affine_matrix.shape == (3, 3):
                    affine_matrix = affine_matrix[:2, :]

                inverse_affine_matrix = cv2.invertAffineTransform(affine_matrix)
                transformed_mask = cv2.warpAffine(transformed_mask, inverse_affine_matrix, (transformed_mask.shape[1], transformed_mask.shape[0]))
                transformed_mask = transformed_mask[..., np.newaxis]

            elif 'HorizontalFlip' in transform_name and t["applied"]:

                transformed_mask = A.Compose([A.HorizontalFlip(p=1)])(image=transformed_mask)['image']
                
            elif 'VerticalFlip' in transform_name and t['applied']:
                transformed_mask = A.Compose([A.VerticalFlip(p=1)])(image=transformed_mask)['image']
        transformed_mask = torch.tensor(np.transpose(transformed_mask, (2, 0, 1)))

        return transformed_mask
    
    def augment_and_predict(self, image, T):

        """
        :param image: Input image (C, H, W)
        :param T: Number of augmentations to perform
        """
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C) for albumentations
        # Store transformed images and masks
        augmented_images = []
        predicted_masks = []
        replay_data_list = []

        for _ in range(T):
            augmented = self.transforms(image=image)
            aug_image = augmented['image']
            aug_image = torch.tensor(np.transpose(aug_image, (2, 0, 1)))
            augmented_images.append(aug_image)
            replay_data_list.append(augmented['replay'])  # Store augmentation params for reverse alignment

            # Predict segmentation mask (add batch dimension)
            with torch.no_grad():
                mask_pred = self.model(aug_image.unsqueeze(0))

            predicted_masks.append(mask_pred)

        # fig, axes = plt.subplots(1, 5, figsize=(20, 8))

        # for i, ax in enumerate(axes.flat):
        #     im = np.transpose(augmented_images[i].numpy(), (1, 2, 0))
        #     im = (im - im.min()) / (im.max() - im.min()) 
        #     ax.imshow(im)
        #     ax.axis('off')
        # plt.show()

        return predicted_masks, replay_data_list


    def compute_entropy(self, masks):
        # Stack along the new dimension to get shape [T, H, W] for pixel-wise computation
        stacked_masks = torch.stack(masks, dim=0)
        probs = torch.sigmoid(stacked_masks)  # Apply softmax if model outputs logits
        entropy_map = entropy(probs.cpu().numpy(), axis=0)
        return torch.tensor(entropy_map)

    def evaluate(self, T = 5):
        self.model.train(False)
        self.model.eval()
        

        allInputs, allGT, allEntropies, allPred = [], [], [], []
        for (images, GTs, resizedImg) in self.testDataLoader:
            images      = images.to(self.device)

            for image in images:
                
                predIm = torch.sigmoid(self.model(image.unsqueeze(0)).squeeze()).detach().numpy()
                predicted_masks, replay_data_list = self.augment_and_predict(image.data.numpy(), T)
   
                inverted_masks = [self.inverse_transform(mask, replay_data) for mask, replay_data in zip(predicted_masks, replay_data_list)]
                entropy_map = self.compute_entropy(inverted_masks)

                allEntropies.append(entropy_map)
                allPred.append(predIm)

            allInputs.extend(resizedImg.data.numpy())
            allGT.extend(GTs.data.numpy())

        allInputs = np.array(allInputs)
        allGT     = np.array(allGT)
        allEntropies = np.array(allEntropies).squeeze()
        allPred = np.array(allPred).squeeze()

        showPredictions(allInputs, allEntropies, allGT, allPred, self.resultsPath)







