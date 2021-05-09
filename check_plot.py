import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress

model_name = "VNet_SSE_DiceLoss_tv_1500images_10batchsize_17_04_2021_11_45" ##

def plot_loss_curves(model_name):
    '''Plot loss curves of trained model'''

    # Get loss files
    script_dir = os.path.dirname(__file__)
    train_file_path = os.path.join(script_dir, ("results/trainLoss_"+model_name))
    val_file_path = os.path.join(script_dir, ("results/valLoss_"+model_name))

    # Open files
    with open(train_file_path, "r") as f:
        trainLoss = [float(value) for value in f.read().split()]
        trainLoss = np.asarray(trainLoss).flatten()
        x_trainLoss = np.arange(trainLoss.shape[0])

    with open(val_file_path, "r") as f:
        valLoss = [float(value) for value in f.read().split()]
        valLoss=np.asarray(valLoss).flatten()
        x_valLoss=np.linspace(0,trainLoss.shape[0],valLoss.shape[0])

    # Plot loss curves
    plt.title('Loss Curves')
    plt.plot(x_trainLoss,trainLoss,label='Training Loss')
    plt.plot(x_valLoss,valLoss,label='Validation Loss')
    plt.legend()
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss Value')

    ## Calculate linear regression of 500 last iterations
    ##y=trainLoss[trainLoss.shape[0]-500:] #[trainLoss.shape[0]//2:]
    ##x= np.arange(trainLoss.shape[0]-500, trainLoss.shape[0]) #(trainLoss.shape[0]//2, trainLoss.shape[0])
    ##slope, intercept, r, p, se = linregress(x, y)
    ##print('slope: '+str(slope))
    ##yreg=slope*x + intercept
    ##
    ##print('mean, 500 last:')
    ##print(np.mean(trainLoss[trainLoss.shape[0]-500:]))
    ##print('mean, others:')
    ##print(np.mean(trainLoss[:trainLoss.shape[0]-500]))
    ## Plot linear regression
    ##plt.plot(x,yreg,'--')

    plt.show()

if __name__=="__main__":
    plot_loss_curves(model_name)