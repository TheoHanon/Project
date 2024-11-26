from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from matplotlib.colors import BoundaryNorm



# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


# --------------------------------------------------------------------------------
# DISPLAY A 5X5 IMAGES FROM A DATALOADER INSTANCE
# INPUTS: 
#     - dataLoader (Dataset): Instance of Dataset
#     - param (dic): dictionnary containing the parameters defined in the 
#                    configuration (yaml) file
# --------------------------------------------------------------------------------
def showDataLoader(dataLoader, param):
    cols, rows = 5, 5
    figure, ax = plt.subplots(nrows=rows, ncols=cols, dpi=280)
    row = 0
    for (imgBatch, maskBatch, _) in dataLoader:
        if row == rows: 
            break
        for col in range(cols): 
            img  = imgBatch[col].numpy().astype('uint8')
            mask = maskBatch[col].numpy().astype('uint8')
            ax[row, col].imshow(img.transpose((1, 2, 0)))
            ax[row, col].imshow(mask*255, interpolation="nearest", alpha=0.3, cmap='Oranges')
            ax[row, col].set_axis_off()
        row += 1 
    plt.tight_layout()
    plt.show()
    

# --------------------------------------------------------------------------------
# DISPLAY A SINGLE IMAGE AND GT MASK WITH THE CORRESPONDING PRECICTION 
# INPUTS: 
#     - img (arr): a 3D numpy array containing the image (ch x depth x width)
#     - pred (arr): a binary 2D numpy array containing the predicted mask
#                  (depth x width)
#     - GT (arr): a binary 2D numpy array containing the ground truth mask
#                  (depth x width)
#     - filePath (str): path to save the image file
# --------------------------------------------------------------------------------
def singlePrediction(img, entropy, GT, pred, filePath): 

    figure = plt.figure(figsize=(10, 10))
    gridspec = figure.add_gridspec(2, 2, wspace=0.02, hspace=0.2)  # Increase vertical spacing slightly

    # Input image
    ax1 = figure.add_subplot(gridspec[0, 0])
    ax1.imshow(img.transpose((1, 2, 0)))
    ax1.set_title("Input", fontweight="bold", fontsize=16, pad=15)  # Increased padding
    ax1.set_axis_off()

    # Ground truth
    ax2 = figure.add_subplot(gridspec[0, 1])
    ax2.imshow(GT, interpolation="nearest")
    ax2.set_title("GT", fontweight="bold", fontsize=16, pad=15)
    ax2.set_axis_off()

    # Predicted probabilities
    ax3 = figure.add_subplot(gridspec[1, 0])
    im = ax3.imshow(entropy, interpolation="nearest", cmap="winter", vmin=0, vmax=1)
    ax3.set_title("Predicted Probabilities", fontweight="bold", fontsize=16, pad=20)  # More padding for bottom row
    ax3.set_axis_off()

    # Predicted mask
    ax4 = figure.add_subplot(gridspec[1, 1])
    ax4.imshow(np.squeeze(pred > 0.5), interpolation="nearest", vmin=0, vmax=1)
    ax4.set_title("Predicted Mask", fontweight="bold", fontsize=16, pad=20)  # More padding for bottom row
    ax4.set_axis_off()

    # Adding the colorbar just below the subplots
    cbar_ax = figure.add_axes([0.21, 0.05, 0.6, 0.02])  # Position closer to subplots
    cbar = figure.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Entropy', fontsize=14)

    # Tighten layout and adjust borders
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust rect for bottom space
    plt.savefig(filePath, bbox_inches='tight')  # Ensure no extra whitespace
    plt.close()

    

# --------------------------------------------------------------------------------
# DISPLAY ALL IMAGES AND PREDICTIONS
# INPUTS: 
#     - allInputs (list): list of 3D numpy arrays containing the input images 
#     - allPreds (list): list of 2D numpy arrays containing the predicted masks
#     - allGT (list): list of 2D numpy arrays containing the ground truth masks 
#     - resultPath (str): path to folder in which to save the image files
# --------------------------------------------------------------------------------
def showPredictions(allInputs, allEntropy, allGT, allPred, resultPath):
    idx = 0
    for (img, entropy, GT, pred) in zip(allInputs, allEntropy, allGT, allPred): 
        filePath = os.path.join(resultPath, "Test", str(idx))
        createFolder(os.path.join(resultPath, "Test"))
        singlePrediction(img, entropy, GT, pred, filePath)
        if idx > 30: 
            break
        idx += 1

    