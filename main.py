'''
File to launch app
'''
# General imports
import sys
sys.path.append("..")
import os
import time
import h5py
from datetime import datetime
from tqdm import tqdm

# Qt imports
from PyQt5 import QtCore
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QTableWidgetItem, QAbstractItemView, \
    QMainWindow, QLabel, QProgressBar, QMessageBox, QProgressDialog

# Calculation imports
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score
import skimage.io as skio
from scipy.ndimage import gaussian_filter, median_filter
import mclahe

# Machine learning imports
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchio as tio
from magic_vnet import vnet
from util import ImageDataset, train, evaluate, DiceLoss, DiceBCELoss, IoULoss, \
    plot, plotSeg, preprocess
from check_plot import plot_loss_curves


class Controller(QMainWindow):
    '''Class for the app's main window'''

    sig_update_progress = QtCore.pyqtSignal(int) #Signal for progress bar in status bar

    def __init__(self):
        QMainWindow.__init__(self)
        
        # Load user interface
        basepath = os.path.join(os.path.dirname(__file__))
        uic.loadUi(os.path.join(basepath,"interface.ui"), self)

        # Create status bar
        self.label_statusBar = QLabel()
        self.progress_statusBar = QProgressBar()
        self.statusbar.addPermanentWidget(self.label_statusBar)
        self.statusbar.addPermanentWidget(self.progress_statusBar)
        self.progress_statusBar.hide()
        self.progress_statusBar.setFixedWidth(250)

        self.sig_update_progress.connect(self.progress_statusBar.setValue)

        # Initiate buttons
        self.pushButton_downloadTraining.clicked.connect(self.select_training_file)
        self.pushButton_trainSegmentation.clicked.connect(self.start_training)
        self.pushButton_showLossCurves.clicked.connect(self.show_loss)

        self.pushButton_downloadTest.clicked.connect(self.select_test_file)
        self.pushButton_downloadTruth.clicked.connect(self.select_truth_file)
        self.pushButton_downloadModel.clicked.connect(self.select_model_file)
        self.pushButton_testSegmentation.clicked.connect(self.start_segmentation)
        self.pushButton_saveSegmentation.clicked.connect(self.save_file)
        self.pushButton_show3DSegmentation.clicked.connect(self.show_3d_segmentation)
        self.pushButton_show2DSegmentation.clicked.connect(self.show_2d_segmentation)
        self.pushButton_showMetrics.clicked.connect(self.show_metrics)

        # Initiate parameters
        self.comboBox_model.insertItems(0,['VNet_SSE','VNet_CSE','VNet_BSC'])
        self.comboBox_lossFunction.insertItems(0,['DiceLoss','DiceBCELoss','IoULoss'])

        self.doubleSpinBox_learningRate.setDecimals(4)
        self.doubleSpinBox_learningRate.setRange(0.0001,1) ##
        self.doubleSpinBox_learningRate.setValue(0.001)
        self.doubleSpinBox_learningRate.setSingleStep(0.0001) ##

        self.doubleSpinBox_nEpochs.setDecimals(0)
        self.doubleSpinBox_nEpochs.setRange(1,100) ##
        self.doubleSpinBox_nEpochs.setValue(10)

        self.doubleSpinBox_nImages.setDecimals(0)
        self.doubleSpinBox_nImages.setRange(1,10000) ##
        self.doubleSpinBox_nImages.setValue(100)

        self.doubleSpinBox_batchSize.setDecimals(0)
        self.doubleSpinBox_batchSize.setRange(1,10) ##
        self.doubleSpinBox_batchSize.setValue(1)

        self.doubleSpinBox_segSlice.setDecimals(0)

        # ---------- init torch ---------------------------
        torch.manual_seed(0) # generate random seed
        np.random.seed(0)
        # select device for machine learning
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('--Running on GPU (via CUDA)--')
        else:
            self.device = torch.device("cpu")
            print('--Running on CPU--')
        #---------------------------------------------------
        self.modelpath=''
        self.file_testing=''
    
    def show_error_popup(self, error=''):
        '''Shows error popup'''
        
        error_popup = QMessageBox()
        error_popup.setWindowTitle('Program Error')
        error_popup.setText('Error while '+error+', please try again')
        error_popup.setIcon(QMessageBox.Warning)
        error_popup.setStandardButtons(QMessageBox.Ok)
        error_popup.setDefaultButton(QMessageBox.Ok)
        error_popup.exec_()

    def show_progress_bar(self, text=''):
        '''Shows the progress bar'''

        self.label_statusBar.setText(text)
        ## TODO : Fix progress bar
        ##self.progress_statusBar.show()
        ##self.sig_update_progress.emit(50)

    def hide_progress_bar(self):
        '''Hides the progress bar'''

        self.label_statusBar.setText('')
        ##self.progress_statusBar.hide()
        ##self.sig_update_progress.emit(0)

    def select_training_file(self):
        '''Allows the selection of a training file'''
        
        try:
            self.file_training = QFileDialog.getOpenFileName(self, 'Choose Directory','','Hierarchical files (*.hdf5 *.h5)')[0]
            if self.file_training != '': #If file directory specified
                self.label_trainingFile.setText(self.file_training)
                self.pushButton_trainSegmentation.setEnabled(True)
            else:
                self.label_trainingFile.setText('-No file selected')
        except:
            self.show_error_popup('downloading training file')
            print('Download error of training file, please try again')
    
    def start_training(self):
        '''Train segmentation'''

        try:
            # Get hyperparameters
            model_name = self.comboBox_model.currentText()
            self.loss_function = self.comboBox_lossFunction.currentText()
            self.learning_rate = self.doubleSpinBox_learningRate.value()
            self.epochs = int(self.doubleSpinBox_nEpochs.value())
            self.length = int(self.doubleSpinBox_nImages.value())
            self.batch_size = int(self.doubleSpinBox_batchSize.value())

            if not os.path.isdir('results'):
                os.mkdir('results')

            # Update status bar
            self.show_progress_bar('Training segmentation...')

            # ---------- Define model to be used -----
            # Get model type
            if model_name == 'VNet_CSE':
                model=vnet.VNet(1,1, num_blocks=[3,3,3,3])
            elif model_name == 'VNet_BSC':
                model=vnet.VNet_BSC(1,1, num_blocks=[3,3,3,3])
            elif model_name == 'VNet_SSE':
                model=vnet.VNet_SSE(1,1, num_blocks=[3,3,3,3])
            # Get model name
            model = model.to(self.device)
            self.model_name=model_name+'_'+str(self.length)+'images_'+str(self.batch_size) \
                +'batchsize_'+datetime.today().strftime("%d_%m_%Y_%H_%M")
            model_save_path = 'results/'+self.model_name
            #---------------------------------------------------
            if self.loss_function == 'IoULoss':
                criterion = IoULoss()
            elif self.loss_function == 'DiceLoss':
                criterion = DiceLoss()
            elif self.loss_function == 'DiceBCELoss':
                criterion = DiceBCELoss()

            train_start_time = time.time() # Get start time

            #----------- Read dataset --------------------------
            dataset=ImageDataset(self.file_training, length=self.length, augment=0)
            ## TODO: Augment dataset and add noise with TORCHIO
            #---------------------------------------------------

            #----------- Training routine -----------------------
            # Define optimizer to adjust model parameters
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate)
            # Define scheduler to adjust learning rate according to epoch
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
                
            best_val_loss = float("inf")
            loss_train=[]
            loss_val=[]
            best_epoch=0
            total_time=0
                
            ##progress_value = 0 #
            ##progress_increment = 100//self.epochs # for progress bar

            for epoch in range(1, self.epochs + 1):
                    
                epoch_start_time = time.time() # Get start time
                    
                # Prepare data : randomly slit it in 2 groups (training and validation)
                train_ratio=.7 # 70% of images for model training; 30% for validation
                idx1=int(len(dataset)*train_ratio)
                train_dataset, val_dataset = random_split(dataset, [idx1,(len(dataset)-idx1)])

                print('Images for training: '+str(len(train_dataset))+' | Images for Validation: '
                    +str(len(val_dataset)))
                print('-' * 89)

                # Load data
                train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
                    
                # Train model
                loss_tr = train(device=self.device,model=model, criterion=criterion, optimizer=optimizer, 
                    scheduler=scheduler, dataset=train_loader, data_length=len(train_dataset),epoch=epoch, 
                    batch_size=self.batch_size)
                loss_train.append(loss_tr) # Add training loss to epoch total
                
                # Validate model
                val_loss, loss_v = evaluate(device=self.device,model=model,criterion=criterion, 
                    dataset=val_loader, data_length=len(val_dataset), batch_size=self.batch_size)
                loss_val.append(loss_v) # Add validation loss to epoch total
                    
                epoch_elapsed = time.time() - epoch_start_time # Calculate epoch training time
                total_time += epoch_elapsed
                # Print epoch training info
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.5f}'
                    .format(epoch, epoch_elapsed, val_loss))
                print(val_loss)
                print('-' * 89)
                
                # save model if best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch=epoch
                    torch.save(model, (model_save_path)) # save model in file
                    self.label_modelFile.setText(model_save_path)

                scheduler.step() # adjust learning rate

                # Update progress bar
                ##progress_value += progress_increment
                ##self.sig_update_progress.emit(int(progress_value))

            ##self.sig_update_progress.emit(100)
            self.modelpath = model_save_path##

            self.train_time = time.time() - train_start_time # calculate train time
            # Print total time
            print('-' * 89)
            print('best_epoch: '+str(best_epoch))
            print('Total training time: {:5.2f}s'.format(total_time))
                
            # Save train/val loss curve
            self.label_savedModel.setText('Model saved at '+model_save_path)
            np.savetxt('results/valLoss_'+self.model_name, np.array(loss_val))#np.savetxt('results/valLoss_'+self.model_name+'.txt', np.array(loss_val))
            np.savetxt('results/trainLoss_'+self.model_name, np.array(loss_train))#np.savetxt('results/trainLoss_'+self.model_name+'.txt', np.array(loss_train))

            # Show training info
            self.show_training_info()

            self.pushButton_showLossCurves.setEnabled(True)

        except:
            self.show_error_popup('training segmentation')
            print('Training error, please try again')

        # Update status bar
        self.hide_progress_bar()

    def show_training_info(self):
        '''Show training info as a table'''

        try:
            # Set axis' names
            self.tableWidget_trainingInfo.setColumnCount(1)
            self.tableWidget_trainingInfo.setRowCount(3) ##
            self.tableWidget_trainingInfo.setHorizontalHeaderItem(0,QTableWidgetItem('Value'))
            self.tableWidget_trainingInfo.setVerticalHeaderItem(0,QTableWidgetItem('Date'))
            self.tableWidget_trainingInfo.setVerticalHeaderItem(1,QTableWidgetItem('Time'))
            self.tableWidget_trainingInfo.setVerticalHeaderItem(2,QTableWidgetItem('Duration'))
            # Set values in table
            date = datetime.today().strftime("%d-%m-%Y")
            hour = datetime.today().strftime("%Hh%M")
            time = '{:5.2f} s'.format(self.train_time)
            self.tableWidget_trainingInfo.setItem(0,0,QTableWidgetItem(date))
            self.tableWidget_trainingInfo.setItem(1,0,QTableWidgetItem(hour))
            self.tableWidget_trainingInfo.setItem(2,0,QTableWidgetItem(time))
            self.tableWidget_trainingInfo.resizeColumnsToContents()
            self.tableWidget_trainingInfo.setEditTriggers(QAbstractItemView.NoEditTriggers) #No editing possible
        except:
            self.show_error_popup('displayin training info')
            print('Display error of training info, please try again')

    def show_loss(self):
        '''Display training loss curves'''

        try:
            self.model_name = self.modelpath[(self.modelpath.find('VNet')):]
            plot_loss_curves(self.model_name) # Plot curves
        except:
            self.show_error_popup('displaying training loss curves')
            print('Display error of training loss curves, please try again')

    def select_test_file(self):
        '''Allows the selection of a test file'''
        
        try:
            self.file_testing = QFileDialog.getOpenFileName(self, 'Choose Directory','','Images (*.tif *.tiff)')[0]
            if self.file_testing != '': #If file directory specified
                self.label_testFile.setText(self.file_testing)
                if self.modelpath != '':
                    self.pushButton_testSegmentation.setEnabled(True)
            else:
                self.label_testFile.setText('-No file selected')
        except:
            self.show_error_popup('downloading segmentation file')
            print('Download error of image file, please try again')

    def select_truth_file(self):
        '''Allows the selection of a ground truth file'''
        
        try:
            self.file_truth = QFileDialog.getOpenFileName(self, 'Choose Directory','','Images (*.tif *.tiff)')[0]
            if self.file_truth != '': #If file directory specified
                self.label_truthFile.setText(self.file_truth)
            else:
                self.label_truthFile.setText('-No file selected')
        except:
            self.show_error_popup('downloading ground truth file')
            print('Download error of image file, please try again')

    def select_model_file(self):
        '''Allows the selection of a model file'''
        
        try:
            self.modelpath = QFileDialog.getOpenFileName(self, 'Choose Directory')[0]
            if self.modelpath != '': #If file directory specified
                self.label_modelFile.setText(self.modelpath)
                self.pushButton_showLossCurves.setEnabled(True)
                if self.file_testing != '':
                    self.pushButton_testSegmentation.setEnabled(True)
            else:
                self.label_modelFile.setText('-No model selected')
                self.pushButton_showLossCurves.setEnabled(False)
        except:
            self.show_error_popup('downloading model file')
            print('Download error of model file, please try again')

    def start_segmentation(self):
        '''Applies trained model to segment image'''

        try:
            # Update status bar
            self.show_progress_bar('Segmenting image...')

            print('Starting segmentation...')
            seg_start_time = time.time() # get start time

            imagepath=self.file_testing
            segpath=self.file_truth
            
            kernel_size=(64,64,64) # Size of the training images

            im=skio.imread(imagepath)
            self.im_origin = im

            if self.checkBox_preprocess.isChecked() == True:
                im = preprocess(im)

            # Augment contrast with adaptive hist equalization
            im=mclahe.mclahe(im, kernel_size=kernel_size, adaptive_hist_range=True)

            s1, s2, s3 = np.array(im.shape).astype(float)
        
            # Modify image size on dim1, dim2 and dim3 to be multiples of 
            # k1, k2, k3 (the size of the training images)
            k1, k2, k3 = np.array(kernel_size).astype(float)
            d1=((k1*(s1%k1>0))-s1%k1)
            d2=((k2*(s2%k2>0))-s2%k2)
            d3=((k3*(s3%k3>0))-s3%k3)
            im=np.pad(im, ((0,int(d1)),(0,int(d2)),(0,int(d3))), constant_values=0) # Add padding
        
            # Check kernel size
            ks1=int(min(s1, k1))
            ks2=int(min(s2, k2))
            ks3=int(min(s2, k3))
        
            # Extract patches of size (ks1, ks2, ks3)
            patches=torch.FloatTensor(im.copy())
            patches=patches.unfold(0, ks1, ks1).unfold(1, ks2, ks2).unfold(2, ks3, ks3)
            unfold_shape=patches.shape
            patches=patches.contiguous().view(-1, ks1, ks2, ks3).unsqueeze(1).unsqueeze(1)
        
            # Run prediction on each patch...
            model=torch.load(self.modelpath)
            model=model.to(self.device)
            patches=patches.to(self.device)
            model.eval() # Set evaluation mode
            with torch.no_grad():
                patches=[model(i) for i in tqdm(patches)] # Apply model to each patch
                patches=torch.stack(patches)
            
            # Stitch back together patches
            output_c = unfold_shape[0] * unfold_shape[3]
            output_h = unfold_shape[1] * unfold_shape[4]
            output_w = unfold_shape[2] * unfold_shape[5]
            out = patches.view(unfold_shape)
            out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
            out = out.view(output_c, output_h, output_w)
            out = torch.sigmoid(out)
        
            # Remove padding
            out = out[:-int(d1),:-int(d2),:-int(d3)]
            im = im[:-int(d1),:-int(d2),:-int(d3)]

            # Apply threshold to probability map
            out = out.cpu().numpy() # 
            threshold = 0.5 ## Arbitrary treshold
            out=(out>threshold).astype('uint8') # Return binary image
            # Retrieve seg ground truth
            seg=(skio.imread(segpath)>0).astype(int)
            seg = np.where(seg==0,0,1)
            if self.checkBox_preprocess.isChecked() == True:
                seg = seg[14:50,:,:400] ## For test with Mask_OCT_test.tiff, else comment out
            else:
                seg = seg[:50,:,:] ## For test with mouse10_seg_50_400.tiff, else comment out
                out = out[:50,:,:] ## For test with mouse10_seg_50_400.tiff, else comment out

            self.im = im
            self.out = out
            self.seg = seg

            self.seg_time = time.time() - seg_start_time # calculate seg time
            ##self.sig_update_progress.emit(100)

            # Show info
            self.show_test_info()

            self.pushButton_saveSegmentation.setEnabled(True)
            self.pushButton_show3DSegmentation.setEnabled(True)
            self.pushButton_show2DSegmentation.setEnabled(True)
            self.doubleSpinBox_segSlice.setMaximum(self.im.shape[0]-1)
            self.pushButton_showMetrics.setEnabled(True)
        except:
            self.show_error_popup('segmenting image')
            print('Segmentation error, please try again')

        # Update status bar
        self.hide_progress_bar()

    def show_test_info(self):
        '''Show segmentation info as a table'''
        
        try:
            # Set axis' names
            self.tableWidget_testInfo.setColumnCount(1)
            self.tableWidget_testInfo.setRowCount(4)
            self.tableWidget_testInfo.setHorizontalHeaderItem(0,QTableWidgetItem('Value'))
            self.tableWidget_testInfo.setVerticalHeaderItem(0,QTableWidgetItem('Date'))
            self.tableWidget_testInfo.setVerticalHeaderItem(1,QTableWidgetItem('Time'))
            self.tableWidget_testInfo.setVerticalHeaderItem(2,QTableWidgetItem('Duration'))
            self.tableWidget_testInfo.setVerticalHeaderItem(3,QTableWidgetItem('Image Shape'))
            # Print values in table
            date = datetime.today().strftime("%d-%m-%Y")
            hour = datetime.today().strftime("%Hh%M")
            time = '{:5.2f} s'.format(self.seg_time)
            self.tableWidget_testInfo.setItem(0,0,QTableWidgetItem(date))
            self.tableWidget_testInfo.setItem(1,0,QTableWidgetItem(hour))
            self.tableWidget_testInfo.setItem(2,0,QTableWidgetItem(time))
            self.tableWidget_testInfo.setItem(3,0,QTableWidgetItem(str(self.im.shape)))
            self.tableWidget_testInfo.resizeColumnsToContents()
            self.tableWidget_testInfo.setEditTriggers(QAbstractItemView.NoEditTriggers) #No editing possible
        except:
            self.show_error_popup('displaying segmentation info')
            print('Display error of segmentation info, please try again')

    def save_file(self):
        '''Allows the saving of a segmentation file'''
        
        try:
            # Select directory
            options = QFileDialog.Options()
            options |= QFileDialog.DontResolveSymlinks
            options |= QFileDialog.ShowDirsOnly
            self.save_directory = QFileDialog.getExistingDirectory(self, 'Choose Directory', '', options)
            if self.save_directory != '': #If directory specified
                # Save file
                self.model_name = self.modelpath[(self.modelpath.find('VNet')):]
                filename = self.model_name ##
                savepath = self.save_directory + '/SEGMENTATION_'+filename+'.tiff'

                print('Saving output: '+savepath)
                skio.imsave(savepath, self.out*255) # save image in a TIF file

                self.label_savedFile.setText(savepath)
            else:
                self.label_savedFile.setText('-Fichier non enregistré')
        except:
            self.show_error_popup('saving segmented image')
            print('Save error of segmented image, please try again')

    def show_3d_segmentation(self):
        '''Shows a 3D segmented image'''
        
        try:
            # Update status bar
            self.show_progress_bar('Plotting 3D segmented image...')

            # plot with mayavi in 3D
            print('Plotting 3D image with mayavi...')
            ##plot(self.im, self.out>0.5)
            plotSeg(self.out>0.5)
        except:
            self.show_error_popup('displaying segmented image in 3D')
            print('Display error of 3D segmented image, please try again')
        
        # Update status bar
        self.hide_progress_bar()

    def show_2d_segmentation(self):
        '''Shows a 2D slice of the segmented image'''
        
        try:
            slice_i = int(self.doubleSpinBox_segSlice.value())
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.title('Original Image, slice '+str(slice_i))
            plt.imshow(self.im_origin[slice_i,:,:], cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title('Preprocessed Image, slice '+str(slice_i))
            plt.imshow(self.im[slice_i,:,:], cmap='gray')
            plt.subplot(2, 2, 3)
            plt.title('Ground Truth, slice '+str(slice_i))
            plt.imshow(self.seg[slice_i,:,:], cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title('Segmentation, slice '+str(slice_i))
            plt.imshow(self.out[slice_i,:,:], cmap='gray')
            plt.tight_layout()
            plt.show(block=False)
        except:
            self.show_error_popup('displaying 2D slice of segmented image')
            print('Display error of 2D slice of segmented image, please try again')

    def show_metrics(self):
        '''Shows metrics applied to a segmented image'''

        try:
            # Update status bar
            self.show_progress_bar('Calculating metrics...')

            time.sleep(1)
            # Caculate metrics
            tn, fp, fn, tp = confusion_matrix(self.seg.flatten(), self.out.flatten()).ravel()
            tn = int(tn)
            fp = int(fp)
            fn = int(fn)
            tp = int(tp)
            
            specificity = tn / (fp+tn)
            sensitivity = tp / (tp+fn)
            accuracy = (tp+tn) / (tp+tn+fp+fn)
            dice = 2*tp / (2*tp+fp+fn)
            jaccard = tp / (tp+fp+fn)
            mcc = (tp*tn-fp*fn) / (math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            ##self.sig_update_progress.emit(100) ##

            metrics = {'Specificity':specificity,'Sensitivity':sensitivity,'Accuracy':accuracy,
            'Dice coefficient':dice,'Jaccard index':jaccard,'Matthews correlation coefficient':mcc}

            # Show metrics
            self.tableWidget_metrics.setColumnCount(2)
            self.tableWidget_metrics.setRowCount(len(metrics))
            self.tableWidget_metrics.setHorizontalHeaderItem(0,QTableWidgetItem('Metric'))
            self.tableWidget_metrics.setHorizontalHeaderItem(1,QTableWidgetItem('Value'))
            for i in range(len(metrics)):
                key,value = list(metrics.items())[i]
                self.tableWidget_metrics.setItem(i,0,QTableWidgetItem(key))
                self.tableWidget_metrics.setItem(i,1,QTableWidgetItem('{:5.3f}'.format(value)))
            self.tableWidget_metrics.resizeColumnsToContents()
            self.tableWidget_metrics.setEditTriggers(QAbstractItemView.NoEditTriggers) #No editing possible
        except:
            self.show_error_popup('calculating segmentation metrics')
            print('Calculation error of segmentation metrics, please try again')

        # Update status bar
        self.hide_progress_bar()

# Launch app
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
app = QApplication(sys.argv)
controller = Controller()
controller.show()
app.exec_()