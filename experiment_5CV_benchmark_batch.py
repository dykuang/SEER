# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:57:24 2020

@author: dykua

For the benchmark purpose - 5cv
"""
#%%
import argparse
from re import sub
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import zscore
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from Utils import scores
from Models import TFCNet_multiWD, EEGNet, DeepConvNet, WD_EEGNet, TFCNet_multiWD_2d, TFCNet_multiWD_2d_abl
#from visual import plot_confusion_matrix
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

#%%
path = '/mnt/HDD/Datasets/SEED'
# path = '/media/dykuang/SATA/Datasets/SEED'
parser = argparse.ArgumentParser()
parser.add_argument('--subject', help='subject index')
parser.add_argument('--chns', default=4, help='channel profile')
parser.add_argument('--nn_choice', default=0, help = 'model type')
args = parser.parse_args()

subject = '{:02d}'.format( int(args.subject) )

# X = loadmat(path + '\S{}_E01.mat'.format(subject))['segs'].transpose([2,1,0])
# Y = loadmat(path + '\Label.mat')['seg_labels'][0]
X = loadmat( os.path.join(path, 'S{}_E01.mat'.format(subject)) )['segs'].transpose([2,1,0])
Y = loadmat( os.path.join(path, 'Label.mat') )['seg_labels'][0]

#chns_choice = 4 # 0: 4 chns, 1: 6 chns, 2: 9 chns, 3: 12 chns, 4: all chns
chns_choice = int(args.chns)

if chns_choice == 0:
    chns = np.array([14, 22, 23, 31])
elif chns_choice == 1:
    chns = np.array([14, 22, 23, 31, 32, 40])
elif chns_choice == 2:
    chns = np.array([0,1,2,14,22,23,31,32,40])
elif chns_choice == 3:
    chns = np.array([14, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 50])
elif chns_choice == 4:
    chns = np.arange(62)

X = X[...,chns]
chns_token = '{:02d}'.format(len(chns))

#nn_choice = 5 # 0: proposed, 1: eegnet, 2: deepconv , 3: Weegnet , 4: proposed - 2d, 5: WD version of (4)
nn_choice = int(args.nn_choice)
if nn_choice == 0:
    nn_token = 'proposed'
    Params = {
        'shape': (X.shape[1], len(chns)),
        'num classes': 3,
        'depth_act': 'tanh',
        'sep_act': 'linear',
        'merge': 'A',
        'WD_spec' : [[8, 5, 1]]*1, # num, kernel length, stride
        'depth multiplier':1,
        'depth rate':1,
        'merge ker num': 8,
        'merge ker len': 5,
        'num_filters_list':[8, 8], 
        'kernel_size_list':[5,5],
        'strides_for_pool':[2,2],   
        'droprate':0.1, 
        'spatial droprate': 0.0,
        'normrate_head':1.0, 
        'normrate_dense':0.25,
        'batchsize': 128, 
        'epochs':80,
        'lr': 1e-2,
        }
    
    model = TFCNet_multiWD(Params['shape'], Params['num classes'], 
                        dep_activation =  Params['depth_act'], sep_activation =  Params['sep_act'],
                        merge_style = Params['merge'], use_WD = False,
                        WDspec_list = Params['WD_spec'], # Number, len, strides
                        depth_multiplier = Params['depth multiplier'], depth_rate=Params['depth rate'], 
                        merge_kernel_num = Params['merge ker num'], merge_kernel_len = Params['merge ker len'],
                        num_filters_list = Params['num_filters_list'], kernel_size_list=Params['kernel_size_list'],
                        strides_for_pool=Params['strides_for_pool'],
                        learning_rate=Params['lr'], droprate=Params['droprate'], 
                        spatial_droprate=Params['spatial droprate'],
                        normrate_head=Params['normrate_head'], 
                        normrate_dense = Params['normrate_dense'])

    
    
elif nn_choice == 1:
    nn_token = 'eegnet'
    model = EEGNet(nb_classes = 3, Chans = len(chns), Samples = 200, 
                   dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                   D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                   learning_rate = 1e-2)
    
elif nn_choice ==2:
    nn_token = 'deepconv'
    model = DeepConvNet(nb_classes= 3, Chans = len(chns), Samples = 200,
                        dropoutRate = 0.25, learning_rate = 1e-2)
    
    
elif nn_choice == 3:
    nn_token = 'wdeeg'
    model = WD_EEGNet(nb_classes = 3, Chans = len(chns), Samples = 200, 
                      WD_spec = [8, 5, 1],
                      dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                      D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                      learning_rate = 1e-2)
    
elif nn_choice == 4:
    nn_token = 'proposed_2d'
    Params = {
        'shape': (X.shape[1], len(chns), 1),
        'num classes': 3,
        'depth_act': 'tanh',
        'sep_act': 'linear',
        'merge': 'A',
        'WD_spec' : [[8, 5, 1]]*1, # num, kernel length, stride
        'depth multiplier':1,
        'depth rate':1,
        'merge ker num': 8,
        'merge ker len': 5,
        'num_filters_list':[8, 8], 
        'kernel_size_list':[5,5],
        'strides_for_pool':[2,2],   
        'droprate':0.0, 
        'spatial droprate': 0.0,
        'normrate_head':1.0, 
        'normrate_dense':0.25,
        'batchsize': 128, 
        'epochs':80,
        'lr': 1e-2,
        }
    
    model = TFCNet_multiWD_2d(Params['shape'], Params['num classes'], 
                            dep_activation =  Params['depth_act'], sep_activation =  Params['sep_act'],
                            merge_style = Params['merge'], use_WD = False,
                            WDspec_list = Params['WD_spec'], # Number, len, strides
                            depth_multiplier = Params['depth multiplier'], depth_rate=Params['depth rate'], 
                            merge_kernel_num = Params['merge ker num'], merge_kernel_len = Params['merge ker len'],
                            num_filters_list = Params['num_filters_list'], kernel_size_list=Params['kernel_size_list'],
                            strides_for_pool=Params['strides_for_pool'],
                            learning_rate=Params['lr'], droprate=Params['droprate'], 
                            spatial_droprate=Params['spatial droprate'],
                            normrate_head=Params['normrate_head'], 
                            normrate_dense = Params['normrate_dense'])
    
elif nn_choice == 5:
    nn_token = 'proposed_2dwd'
    Params = {
        'shape': (X.shape[1], len(chns), 1),
        'num classes': 3,
        'depth_act': 'tanh',
        'sep_act': 'linear',
        'merge': 'A',
        'WD_spec' : [[8, 5, 1]]*1, # num, kernel length, stride
        'depth multiplier':1,
        'depth rate':1,
        'merge ker num': 8,
        'merge ker len': 5,
        'num_filters_list':[8, 8], 
        'kernel_size_list':[5,5],
        'strides_for_pool':[2,2],   
        'droprate':0.1, 
        'spatial droprate': 0.0,
        'normrate_head':1.0, 
        'normrate_dense':0.25,
        'batchsize': 128, 
        'epochs':80,
        'lr': 1e-2,
        }
    
    model = TFCNet_multiWD_2d(Params['shape'], Params['num classes'], 
                            dep_activation =  Params['depth_act'], sep_activation =  Params['sep_act'],
                            merge_style = Params['merge'], use_WD = True,
                            WDspec_list = Params['WD_spec'], # Number, len, strides
                            depth_multiplier = Params['depth multiplier'], depth_rate=Params['depth rate'], 
                            merge_kernel_num = Params['merge ker num'], merge_kernel_len = Params['merge ker len'],
                            num_filters_list = Params['num_filters_list'], kernel_size_list=Params['kernel_size_list'],
                            strides_for_pool=Params['strides_for_pool'],
                            learning_rate=Params['lr'], droprate=Params['droprate'], 
                            spatial_droprate=Params['spatial droprate'],
                            normrate_head=Params['normrate_head'], 
                            normrate_dense = Params['normrate_dense'])

elif nn_choice == 6:
    nn_token = 'proposed_2d_S'   #only using the S branch
    Params = {
        'shape': (X.shape[1], len(chns), 1),
        'num classes': 3,
        'depth_act': 'tanh',
        'sep_act': 'linear',
        'merge': 'A',
        'WD_spec' : [[8, 5, 1]]*1, # num, kernel length, stride
        'depth multiplier':1,
        'depth rate':1,
        'merge ker num': 8,
        'merge ker len': 5,
        'num_filters_list':[8, 8], 
        'kernel_size_list':[5,5],
        'strides_for_pool':[2,2],   
        'droprate':0.0, 
        'spatial droprate': 0.0,
        'normrate_head':1.0, 
        'normrate_dense':0.25,
        'batchsize': 128, 
        'epochs':80,
        'lr': 1e-2,
        }
    
    model = TFCNet_multiWD_2d_abl(Params['shape'], Params['num classes'], 
                            dep_activation =  Params['depth_act'], sep_activation =  Params['sep_act'],
                            merge_style = Params['merge'], use_WD = False,
                            WDspec_list = Params['WD_spec'], # Number, len, strides
                            depth_multiplier = Params['depth multiplier'], depth_rate=Params['depth rate'], 
                            merge_kernel_num = Params['merge ker num'], merge_kernel_len = Params['merge ker len'],
                            num_filters_list = Params['num_filters_list'], kernel_size_list=Params['kernel_size_list'],
                            strides_for_pool=Params['strides_for_pool'],
                            learning_rate=Params['lr'], droprate=Params['droprate'], 
                            spatial_droprate=Params['spatial droprate'],
                            normrate_head=Params['normrate_head'], 
                            normrate_dense = Params['normrate_dense'],
                            mode = 'S')

elif nn_choice == 7:
    nn_token = 'proposed_2d_C'   #only using the C branch
    Params = {
        'shape': (X.shape[1], len(chns), 1),
        'num classes': 3,
        'depth_act': 'tanh',
        'sep_act': 'linear',
        'merge': 'A',
        'WD_spec' : [[8, 5, 1]]*1, # num, kernel length, stride
        'depth multiplier':1,
        'depth rate':1,
        'merge ker num': 8,
        'merge ker len': 5,
        'num_filters_list':[8, 8], 
        'kernel_size_list':[5,5],
        'strides_for_pool':[2,2],   
        'droprate':0.0, 
        'spatial droprate': 0.0,
        'normrate_head':1.0, 
        'normrate_dense':0.25,
        'batchsize': 128, 
        'epochs':80,
        'lr': 1e-2,
        }
    
    model = TFCNet_multiWD_2d_abl(Params['shape'], Params['num classes'], 
                            dep_activation =  Params['depth_act'], sep_activation =  Params['sep_act'],
                            merge_style = Params['merge'], use_WD = False,
                            WDspec_list = Params['WD_spec'], # Number, len, strides
                            depth_multiplier = Params['depth multiplier'], depth_rate=Params['depth rate'], 
                            merge_kernel_num = Params['merge ker num'], merge_kernel_len = Params['merge ker len'],
                            num_filters_list = Params['num_filters_list'], kernel_size_list=Params['kernel_size_list'],
                            strides_for_pool=Params['strides_for_pool'],
                            learning_rate=Params['lr'], droprate=Params['droprate'], 
                            spatial_droprate=Params['spatial droprate'],
                            normrate_head=Params['normrate_head'], 
                            normrate_dense = Params['normrate_dense'],
                            mode = 'C')

print('#'*50)
print('CV on model {} with {} channels for subject {}.'.format(nn_token, chns_token, subject)) 
print('#'*50)
       
model.summary()
model.save_weights('model_ini.h5') # save an intial copy to reload at each fold

#%%
from sklearn.model_selection import StratifiedKFold, train_test_split

# using the same validation set
indices = np.arange(len(Y))
_, _, Ycv, Yval, CV_ind, val_ind = train_test_split(X[...,0], Y, indices, test_size=0.1667, 
                                                    random_state=532, shuffle=True, stratify = Y)
Xcv = X[CV_ind]
Xval = X[val_ind]

Xval_transformed = zscore(Xval, axis=1)
if nn_choice in [1, 2, 4]:
    Xval_transformed = Xval_transformed[...,None]
Yval_OH = to_categorical(Yval+1, 3)

# 5 -fold cv 
skf = StratifiedKFold(n_splits=5, shuffle = True, random_state=321)
indexes = skf.split(Xcv, Ycv)


fold_count = 0 
summary = []
ConM = []
for train_index, test_index in indexes:
    print('Fold {} started.'.format(fold_count))
    
    Xtrain, Xtest = Xcv[train_index], Xcv[test_index]
    Ytrain, Ytest = Ycv[train_index], Ycv[test_index]
    
    '''
    Normalize
    '''
    X_train_transformed = zscore(Xtrain, axis=1)
    X_test_transformed = zscore(Xtest, axis=1)
    
    if nn_choice in [1, 2]:
        X_train_transformed = X_train_transformed[...,None]
        X_test_transformed = X_test_transformed[...,None]
    
    Ytrain_OH = to_categorical(Ytrain+1, 3)
    Ytest_OH = to_categorical(Ytest+1, 3)
    
    #%% Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.75, patience=10, min_lr=1e-4)
    #if path.startswith('/media'):
    #    cpt_path = '/media/dykuang/SATA/Benchmarks/SEED/tmp/S{}_checkpoint_{}_{}chns_fold{}'.format(subject, nn_token, chns_token, fold_count)
    #else:
    #    cpt_path = r'E:\Benchmarks\SEED\tmp\S{}_checkpoint_{}_{}chns_fold{}_nodrop'.format(subject, nn_token, chns_token, fold_count)
    cpt_path = '/mnt/HDD/temp/SEER_ckpt/S{}_checkpoint_{}_{}chns_fold{}'.format(subject, nn_token, chns_token, fold_count)    
    cpt = ModelCheckpoint(filepath=cpt_path,
                          save_weights_only=True,
                          monitor='val_accuracy',
                          mode='max',
                          save_best_only=True)
    
    #%% Training
    model.load_weights('model_ini.h5') # each fold starting with the same initialization
    model.fit(X_train_transformed, Ytrain_OH, 
              epochs=80, batch_size = 128,
              # validation_split=0.3,
              validation_data = (Xval_transformed, Yval_OH),
              verbose=1,
              callbacks=[reduce_lr, cpt],
              shuffle = True
             )

    model.load_weights(cpt_path)
    pred = model.predict(X_test_transformed)
    CM = confusion_matrix(Ytest+1, np.argmax(pred, axis=1))
    print(CM)
    
    _, b = scores(CM )
    print(b)
    summary.append(b)
    ConM.append( CM )
    
    
    print('Fold {} finished.'.format(fold_count))   
    print('#'*40)
    fold_count += 1

summary = np.array(summary)
#if path.startswith('/media'):
#    np.save('/media/dykuang/SATA/Benchmarks/SEED/benchmark_summary/S{}_{}_{}chns_nodrop'.format(subject, nn_token, chns_token), summary)
#    np.save('/media/dykuang/SATA/Benchmarks/SEED/benchmark_summary/CM_S{}_{}_{}chns_nodrop'.format(subject, nn_token, chns_token), ConM)
#else:
np.save(r'/mnt/HDD/Benchmarks/SEED/SEER/S{}_{}_{}chns'.format(subject, nn_token, chns_token), summary)
np.save(r'/mnt/HDD/Benchmarks/SEED/SEER/CM_S{}_{}_{}chns'.format(subject, nn_token, chns_token), ConM)


print('mean: {}'.format(np.mean(summary, axis = 0)))
print('std: {}'.format(np.std(summary, axis = 0))) 


# total_CM = ConM[0] + ConM[1] + ConM[2]
# plot_confusion_matrix(total_CM, ['Negative', 'Neutral', 'Positive'], True)
# %%
