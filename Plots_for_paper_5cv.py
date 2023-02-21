# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 21:14:14 2020

@author: dykua

Prepare plots for the paper

syntax changed for tf-keras-vis ver 0.8.0

use new 5cv data
"""
#%%

import tensorflow as tf
# tf.compat.v1.enable_eager_execution() 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.metrics import confusion_matrix
from Utils import scores
from Models import  EEGNet, TFCNet_multiWD_2d
from visual import plot_confusion_matrix


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

#%%

# path = '/media/dykuang/SATA/Datasets/SEED'
# st_path = '/media/dykuang/SATA/Benchmarks/SEED'
path = r'E:\Datasets\SEED'
st_path = r'E:\Benchmarks\SEED'

subject = '15'
#%%
X = loadmat( os.path.join(path, 'S{}_E01.mat'.format(subject)) )['segs'].transpose([2,1,0])

chns_choice = 4 # 0: 4 chns, 1: 6 chns, 2: 9 chns, 3: 12 chns, 4: all chns

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
Y = loadmat( os.path.join(path, 'Label.mat') )['seg_labels'][0]

#%%

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
    'epochs':40,
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


model_wd = TFCNet_multiWD_2d(Params['shape'], Params['num classes'], 
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
    
    

eegnet = EEGNet(nb_classes = 3, Chans = len(chns), Samples = 200, 
               dropoutRate = 0.5, kernLength = 5, F1 = 8, 
               D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
               learning_rate = 1e-2)
    

#%% 
'''
Loading weights
'''   
fold_count = 0
cpt_path = os.path.join( st_path, 'tmp', 'S{}_checkpoint_proposed_2d_{}chns_fold{}'.format(subject, chns_token, fold_count) )
# cpt_path = r'C:\Users\dykua\github\EEG-decoding\tmp\S{}_checkpoint_proposed_2d_{}chns_fold{}'.format(subject, chns_token, fold_count)
model.load_weights(cpt_path)

# cpt_path_wd = r'C:\Users\dykua\github\EEG-decoding\tmp\S{}_checkpoint_proposed_2dwd_{}chns_fold{}'.format(subject, chns_token, fold_count)
# model_wd.load_weights(cpt_path_wd)

# if subject == '03':
#     eegnet_path = r'C:\Users\dykua\github\EEG-decoding\tmp\checkpoint_eegnet_{}chns_fold{}'.format(chns_token, fold_count)
# else:
#     eegnet_path = r'C:\Users\dykua\github\EEG-decoding\tmp\S{}_checkpoint_eegnet_{}chns_fold{}'.format(subject, chns_token, fold_count)

eegnet_path = os.path.join( st_path, 'tmp', 'S{}_checkpoint_eegnet_{}chns_fold{}'.format(subject, chns_token, fold_count) )
eegnet.load_weights(eegnet_path)

#%% saving the channel attention map
# =============================================================================
# from scipy.io import savemat
# save_name = os.path.join(st_path, 'AM_S{}_{}chns_fold{}.mat'.format(subject, chns_token, fold_count) )
# cm = model.get_layer('C_Dep_0').get_weights()[0][0,...,0]
# wm = model.get_layer('F_Dep_0').get_weights()[0][0,...,0]
# savemat(save_name, {'CM': cm, 'WM': wm})

# save_name = os.path.join(st_path, 'EEGNET_AM_S{}_{}chns_fold{}.mat'.format(subject, chns_token, fold_count) )
# # cm = eegnet.get_layer('depthwise_conv2d').get_weights()[0][0,...,0]
# cm = eegnet.layers[3].get_weights()[0][0,...,0]
# savemat(save_name, {'CM': cm})
# =============================================================================

#%% Activation Maximization
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Progress
# from tf_keras_vis.utils.losses import CategoricalScore, SmoothedCategoricalScore
from tf_keras_vis.utils.input_modifiers import Jitter, Rotate
from tf_keras_vis.utils.regularizers import L2Norm, TotalVariation
# from tf_keras_vis.activation_maximization.regularizers import TotalVariation2D, Norm
from matplotlib import cm

def loss(output):
    return (output[0][0], output[1][1], output[2][2])
def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m
def AM_map(model, model_modifier, loss, seed_input, steps = 1000,
           input_modifiers = None, regularizers = None,
           optimizer = tf.optimizers.Adam(1e-2, 0.95), 
           gradient_modifier=None):

    _min = np.amin(seed_input)
    _max = np.amax(seed_input)
    AM = ActivationMaximization(model,
                                model_modifier,
                                clone=False)
                     
    am = AM(
            score = loss,
            # loss = tf.keras.losses.categorical_crossentropy,
            seed_input=seed_input,
            input_range = (_min, _max),
            input_modifiers = input_modifiers,
            regularizers = regularizers,
            steps = steps,
            optimizer = optimizer,
            gradient_modifier=gradient_modifier,
            # callbacks=[Progress()],
            callbacks = None,
            training=False
            )
       
    return am



#%%
selection = 100

ind_list = [np.where(Y==i)[0][selection] for i in range(-1,2,1)]
X_transformed = zscore(X, axis = 1)
Xs = np.array([X_transformed[i] for i in ind_list])
seed_input = tf.random.uniform((3, 200, 62, 1), -2, 2)

eegnet_am = AM_map(eegnet, model_modifier, loss = loss,
                    # seed_input=np.array([X_transformed[i][...,None] for i in ind_list]), 
                    seed_input=seed_input,
                    steps = 500,
                    # input_modifiers = [Jitter(1)], 
                    regularizers = [TotalVariation(10.), L2Norm(10.)],
                    optimizer = tf.optimizers.Adam(1, 0.75),
                    gradient_modifier=None)
# check if make sense
print(eegnet.predict(eegnet_am))

am = AM_map(model, model_modifier, loss = loss, 
            seed_input = seed_input,
            # seed_input = seed_input,
            steps = 500,
            # input_modifiers = None, regularizers = None,
            input_modifiers =[Jitter(2)], 
            regularizers = [TotalVariation(10.), L2Norm(10.)],
            optimizer = tf.optimizers.Adam(1, 0.75),
            gradient_modifier=None)

print(model.predict(am))

am_to_plots = [Xs, eegnet_am[...,0], am[...,0]]
# plt.figure(figsize=(12,8))
f, ax = plt.subplots(3,3, figsize=(12,4))
for i, img in enumerate(am_to_plots):
    for j in range(3):
        mp = np.uint8( cm.summer(img[j].transpose())[..., :3] * 255 )
        ax[j][i].imshow(mp) 
        ax[j][i].set_xticks([])
        ax[j][i].set_yticks(np.arange(0,62, 10))
    ax[2][i].set_xticks(np.arange(0,220,40))  
    ax[2][i].set_xticklabels(['{:.01f}'.format(t) for t in np.arange(0,1.2,0.2)])
plt.subplots_adjust(wspace=0.0, hspace=0.00)
plt.tight_layout() 

#%% AM for the first conv layer 
from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
# layer_name = 'WD-0'
# layer_name = 'conv2d_3'
seed_input = tf.random.uniform((8, 200, 62, 1), 0, 255)
filter_numbers = [i for i in range(8)]

def conv_am(model, layer_name, filter_numbers, seed_input, steps = 100,
            optimizer = tf.optimizers.Adam(1e-2, 0.95), 
            gradient_modifier=None, callbacks = None):
    # This instance constructs new model whose output is replaced to `block5_conv3` layer's output.
    extract_intermediate_layer = ExtractIntermediateLayer(index_or_name=layer_name)
    # This instance modify the model's last activation function to linear one.
    replace2linear = ReplaceToLinear()

    # filter_number = 1
    # def conv_am_score(output):
    #     return output[..., filter_number]
    conv_scores = CategoricalScore(filter_numbers)

    activation_maximization = ActivationMaximization(model,
                                                    # Please note that `extract_intermediate_layer` has to come before `replace2linear`.
                                                    model_modifier=[extract_intermediate_layer, replace2linear],
                                                    clone=False)
    # Generate maximized activation
    activations = activation_maximization(score=conv_scores,
                                          seed_input=seed_input,
                                          steps = steps,
                                          callbacks=callbacks)
    
    return activations

## Since v0.6.0, calling `astype()` is NOT necessary.
# activations = activations[0].astype(np.uint8)

# # Render single
# f, ax = plt.subplots(figsize=(4, 4))
# ax.imshow(activations[0][...,0])
# ax.set_title('filter[{:03d}]'.format(filter_number), fontsize=16)
# ax.axis('off')
# plt.tight_layout()
# plt.show()

'''
The proposed model
'''
activations = conv_am(model, 'WD-0', filter_numbers, seed_input,  
                      steps = 50,
                      optimizer = tf.optimizers.Adam(1, 0.9) )
# Render multiple
f, ax = plt.subplots(nrows=1, ncols=8, figsize=(12, 4))
for i, filter_number in enumerate(filter_numbers):
    ax[i].set_title('filter[{:03d}]'.format(filter_number), fontsize=16)
    ax[i].imshow(activations[i][...,0], cmap = 'nipy_spectral')
    ax[i].axis('off')
plt.tight_layout()
plt.show()

'''
The eegnet
'''
eegnet_activations = conv_am(eegnet, 2, filter_numbers, seed_input,
                             steps = 50,
                             optimizer = tf.optimizers.Adam(1, 0.9))
# Render multiple
f, ax = plt.subplots(nrows=1, ncols=8, figsize=(12, 4))
for i, filter_number in enumerate(filter_numbers):
    ax[i].set_title('filter[{:03d}]'.format(filter_number), fontsize=16)
    ax[i].imshow(eegnet_activations[i][...,0], cmap = 'nipy_spectral')
    ax[i].axis('off')
plt.tight_layout()
plt.show()

'''
One compared
'''
conv_am_to_plots = [eegnet_activations[...,0], activations[...,0]]
# plt.figure(figsize=(12,8))
f, ax = plt.subplots(8,2, figsize=(12,14))
for i, img in enumerate(conv_am_to_plots):
    for j in range(8):
        # mp = np.uint8( cm.jet(img[j].transpose())[..., :3] * 255 )
        mp = img[j].transpose()
        ax[j][i].imshow(mp, cmap='nipy_spectral') 
        ax[j][i].set_xticks([])
        ax[j][i].set_yticks(np.arange(0,62, 10))
    ax[2][i].set_xticks(np.arange(0,220,40))  
    ax[2][i].set_xticklabels(['{:.01f}'.format(t) for t in np.arange(0,1.2,0.2)])
plt.subplots_adjust(wspace=0.0, hspace=0.00)
plt.tight_layout() 


#%% GRAD CAM ++
# from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import GradcamPlusPlus

def gradcam_map(model, model_modifier, loss, seed_input, 
                penultimate_layer,
                # steps = 1000,
                # input_modifiers = None, regularizers = None,
                # optimizer = tf.optimizers.Adam(1e-2, 0.95), 
                # gradient_modifier=None
                ):
    
    gradcam = GradcamPlusPlus(model,
                              model_modifier,
                              clone=False)

    cam = gradcam(score = loss,
                  seed_input = seed_input,
                  penultimate_layer=penultimate_layer, # model.layers number
                  seek_penultimate_conv_layer=True
                
                 )
    # cam = normalize(cam)
    
    return cam

# selection = 100
# ind_list = [np.where(Y==i)[0][selection] for i in range(-1,2,1)]
# Xs = np.array([X_transformed[i] for i in ind_list])

# eegnet
eegnet_cam = gradcam_map(eegnet, model_modifier, loss, 
                         seed_input = np.array([X_transformed[i][...,None] for i in ind_list]), 
                         # seed_input = eegnet_am, 
                         penultimate_layer = 8,
                        #  steps = 1000,
                        #  input_modifiers = None, regularizers = [TotalVariation(10.), L2Norm(10.)],
                        #  optimizer = tf.optimizers.Adam(1, 0.75), 
                        #  gradient_modifier=None
                        )



# propsed
cam =  gradcam_map(model, model_modifier, loss, 
                   seed_input = np.array([X_transformed[i][...,None] for i in ind_list]), 
                   penultimate_layer = -9,
                #    steps = 1000,
                #    input_modifiers = None, regularizers = [TotalVariation(10.), L2Norm(10.)],
                #    optimizer = tf.optimizers.Adam(1, 0.75), 
                #    gradient_modifier=None
                )   

cam_to_plots = [eegnet_cam, cam]

# save the cam result per fold
np.save(r'E:\Benchmarks\SEED\gradcam_fold{}.npy'.format(fold_count), cam)
np.save(r'E:\Benchmarks\SEED\eegnet_gradcam_fold{}.npy'.format(fold_count), eegnet_cam)

f, ax = plt.subplots(3,2,figsize=(8,4))
for i, img in enumerate(cam_to_plots):
    for j in range(3):
        ax[j][i].imshow( np.uint8(cm.summer(Xs[j].transpose())[..., :3] * 255) )
        heatmap = np.uint8(cm.jet(img[j])[..., :3] * 255)
        ax[j][i].imshow(heatmap.transpose((1,0,2)), alpha=0.6)
        ax[j][i].set_xticks([])
        ax[j][i].set_yticks(np.arange(0,62, 10))
    ax[2][i].set_xticks(np.arange(0,220,40))  
    ax[2][i].set_xticklabels(['{:.01f}'.format(t) for t in np.arange(0,1.2,0.2)])
plt.subplots_adjust(wspace=0.0, hspace=0.00)
plt.tight_layout()

# a different visualization
f, ax = plt.subplots(3,3,figsize=(8,4))
for j in range(3):
    ax[0][j].imshow( np.uint8(cm.summer(Xs[j].transpose())[..., :3] * 255) )
    ax[0][j].set_yticks(np.arange(0,62, 10))
    ax[0][j].set_xticks([])
    for i, img in enumerate(cam_to_plots):
        ax[i+1][j].plot(img[j,:,0])
        ax[i+1][j].set_xticks([])
    ax[2][j].set_xticks(np.arange(0,220,40))  
    ax[2][j].set_xticklabels(['{:.01f}'.format(t) for t in np.arange(0,1.2,0.2)])
plt.subplots_adjust(wspace=0.0, hspace=0.00)
plt.tight_layout()

# plot a summary for all the folds
# gcam_eeg = []
# gcam = []
# for i in range(5):
#     gcam_eeg.append(np.load(r'E:\Benchmarks\SEED\eegnet_gradcam_fold{}.npy'.format(i))[:,:,0])
#     gcam.append(np.load(r'E:\Benchmarks\SEED\gradcam_fold{}.npy'.format(i))[:,:,0])

# gcam_eeg =  np.array(gcam_eeg)
# gcam = np.array(gcam)

# m_gcam_eeg = np.mean(gcam_eeg, axis = 0)
# m_gcam= np.mean(gcam, axis = 0)

# std_gcam_eeg = np.std(gcam_eeg, axis = 0)
# std_gcam= np.std(gcam, axis = 0)

# gradcam_all_m = [m_gcam_eeg, m_gcam]
# gradcam_all_std =  [std_gcam_eeg, std_gcam]
# f, ax = plt.subplots(3,3,figsize=(8,4))
# for j in range(3):
#     ax[0][j].imshow( np.uint8(cm.summer(Xs[j].transpose())[..., :3] * 255) )
#     ax[0][j].set_yticks(np.arange(0,62, 10))
#     ax[0][j].set_xticks([])
#     for i, img in enumerate(gradcam_all_m):
#         ax[i+1][j].plot(img[j], label='Mean')
#         ax[i+1][j].fill_between(np.arange(img.shape[1]), 
#                                 gradcam_all_m[i][j]- gradcam_all_std[i][j], 
#                                 gradcam_all_m[i][j]+ gradcam_all_std[i][j] , color="pink", 
#                                 alpha=0.5, label = 'Std')
#         ax[i+1][j].set_ylim([0, 1])
#         ax[i+1][j].set_xticks([])
#     ax[2][j].set_xticks(np.arange(0,220,40))  
#     ax[2][j].set_xticklabels(['{:.01f}'.format(t) for t in np.arange(0,1.2,0.2)])
# plt.subplots_adjust(wspace=0.0, hspace=0.00)
# plt.tight_layout()
# plt.legend()

#%% Saliency
from tf_keras_vis.saliency import Saliency
def saliency_map(model, model_modifier, loss, seed_input, 
                 smooth_samples=20,
                 smooth_noise=0.20,
                 keepdims=False):
    
    saliency = Saliency(model, model_modifier,  clone=False)

    saliency_map = saliency(score = loss,
                            seed_input = seed_input,
                            smooth_samples=smooth_samples,
                            smooth_noise=smooth_noise,
                            keepdims=False
                             )
    # saliency_map = normalize(saliency_map)
    
    return saliency_map

eegnet_sm = saliency_map(eegnet, model_modifier, loss, 
                         seed_input = np.array([X_transformed[i][...,None] for i in ind_list]), 
                         smooth_samples=20,
                         smooth_noise=0.20,
                         keepdims=False)

# f, ax = plt.subplots(3,1)
# for i, title in enumerate(['Sad', 'Neutral', 'Happy']):
#     ax[i].set_title(title, fontsize=14)
#     ax[i].imshow(eegnet_sm[i].transpose(), cmap='jet') 
# plt.tight_layout()

sm = saliency_map(model, model_modifier, loss, 
                  seed_input = np.array([X_transformed[i][...,None] for i in ind_list]),
                  smooth_samples=20,
                  smooth_noise=0.20,
                  keepdims=False)

sm_to_plots = [eegnet_sm, sm]
f, ax = plt.subplots(3,2)
for i, img in enumerate(sm_to_plots):
    for j in range(3):
        ax[j][i].imshow(img[j].transpose(), cmap='jet') 
        ax[j][i].set_xticks([])
        ax[j][i].set_yticks(np.arange(0,62, 10))
    ax[2][i].set_xticks(np.arange(0,220,40))  
    ax[2][i].set_xticklabels(['{:.01f}'.format(t) for t in np.arange(0,1.2,0.2)])
plt.subplots_adjust(wspace=0.0, hspace=0.00)
plt.tight_layout()


#%%
'''
SHAP Explainer

not working for tf > 2.2
'''
# import shap

# # select a set of background examples to take an expectation over
# background = X_transformed[np.random.choice(X_transformed.shape[0], 100, replace=False)]

# # explain predictions of the model on four images
# e = shap.DeepExplainer(eegnet, background)
# # ...or pass tensors directly
# # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
# shap_values = e.shap_values(X_transformed[1:5])

# # plot the feature attributions
# shap.image_plot(shap_values, -X_transformed[1:5])

#%%    

import seaborn as sns
import pandas as pd

#==============================================================================
# A grouped boxplot
#==============================================================================

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color = color, facecolor = color, linewidth=2)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
#    plt.setp(bp['medians'], color=color)
    plt.setp(bp['medians'], linewidth=2)
    plt.setp(bp['fliers'], markersize=4)

def show_box_group(data, names, ticks, colors, box_width = 0.3, sparsity = 3, ymin=0, ymax = 1):
    plt.figure()
    for i, sample in enumerate(data):
        bp = plt.boxplot(sample, positions=np.array(np.arange(sample.shape[1]))*sparsity-0.6+0.4*i,  widths=box_width, sym = 'o',
                  notch=True, patch_artist=True)
        set_box_color(bp, colors[i])
        for patch in bp['boxes']:
           patch.set_alpha(0.8)
        plt.plot([], c=colors[i], label=names[i])
    plt.legend(loc='upper right')

    plt.xticks(np.arange(0, len(ticks) * sparsity, sparsity), ticks, rotation = 45)
    plt.xlim(-2, len(ticks)*sparsity-0.4)
    plt.ylim(ymin, ymax)
    # plt.ylabel('Dice Score')
    #plt.title('Different methods on selected regions')
    plt.grid()
    plt.tight_layout()

summary_all = []
summary_all_std = []
for j in ['deepconv','eegnet', 'proposed_2d']:
    summary_M = []
    summary_M_std = []
    for i in range(1,16):
        temp = np.load(os.path.join(st_path, 'benchmark_summary','S{:02d}_{}_62chns.npy'.format(i, j) ) )
        temp_mean = np.mean(temp, axis=0)
        temp_std = np.std(temp, axis = 0)
        summary_M.append(temp_mean[[0,1,3,4]])
        summary_M_std.append(temp_std[[0,1,3,4]])
    summary_all.append(np.array(summary_M))  
    summary_all_std.append(np.array(summary_M_std))

ticks = ['Acc.', 'Prec.', 'Spec.', 'F1']
colors = ['#2C7BB6', '#2ca25f', '#636363']
box_width = 0.3
sparsity = 3 
show_box_group(summary_all , ['DEEPCONV', 'EEGNET', 'SEER-net'], ticks, colors, ymin=0.8, ymax=1.0)

#%%
'''
Some statistical test
'''

#=========
# Welch's one sided t-test
#===========
from scipy.stats import ttest_ind

def welch_t(subject, path, model_token, chns):
    ac_1 = np.load(os.path.join(path, 'S{:02d}_{}_{}chns.npy'.format(subject, model_token[0], chns) ) )
    ac_2 = np.load(os.path.join(path, 'S{:02d}_{}_{}chns.npy'.format(subject, model_token[1], chns) ) )

    return ttest_ind(ac_1[:,0], ac_2[:,0], equal_var = False)

per_sub_record = []
for i in range(1, 16):
    temp = welch_t(i, os.path.join(st_path, 'benchmark_summary'), ['eegnet', 'proposed_2d'], 62)
    if temp.statistic < 0:
        per_sub_record.append([temp.statistic, 0.5*temp.pvalue])
    else:
        per_sub_record.append([temp.statistic, 1-0.5*temp.pvalue])

#=======
# F-test 
#==========
import scipy.stats as st

def f_test(x, y, alt="two_sided"):
    """
    Calculates the F-test.
    :param x: The first group of data
    :param y: The second group of data
    :param alt: The alternative hypothesis, one of "two_sided" (default), "greater" or "less"
    :return: a tuple with the F statistic value and the p-value.
    """
    df1 = len(x) - 1
    df2 = len(y) - 1
    f = x.var() / y.var()
    if alt == "greater":
        p = 1.0 - st.f.cdf(f, df1, df2)
    elif alt == "less":
        p = st.f.cdf(f, df1, df2)
    else:
        # two-sided by default
        # Crawley, the R book, p.355
        p = 2.0*(1.0 - st.f.cdf(f, df1, df2))
    return f, p

f_test(summary_all[-1][:,0], summary_all[1][:,0], 'less')

# %%
'''
WD v.s Conv2d
'''
from visual import plot_confusion_matrix
from tensorflow.keras.models import Model

def normalize(x):
    return x - np.amin(x)/(np.amax(x) - np.amin(x))


# loading weights
fold_count = 0
subject_for_wd = 1
cpt_path = os.path.join( st_path, 'tmp', 'S{:02d}_checkpoint_proposed_2d_{}chns_fold{}'.format(subject_for_wd, chns_token, fold_count) )
# cpt_path = r'C:\Users\dykua\github\EEG-decoding\tmp\S{}_checkpoint_proposed_2d_{}chns_fold{}'.format(subject, chns_token, fold_count)
model.load_weights(cpt_path)

cpt_path_wd = os.path.join( st_path, 'tmp', 'S{:02d}_checkpoint_proposed_2dwd_{}chns_fold{}'.format(subject_for_wd, chns_token, fold_count) )
model_wd.load_weights(cpt_path_wd)


# prediction after the time convolution
get_wd_f = Model(model_wd.input, model_wd.get_layer('WD-0').output)
get_cv_f = Model(model.input, model.get_layer('WD-0').output)

wd_f = get_wd_f(np.array([X_transformed[i][...,None] for i in ind_list])).numpy().transpose((0,1,3,2))
cv_f = get_cv_f(np.array([X_transformed[i][...,None] for i in ind_list])).numpy()

vis_chns = np.array([14, 22, 23, 31, 32, 40])
wd_vis = wd_f[...,vis_chns,:].reshape((3,200,48))
cv_vis = cv_f[...,vis_chns,:].reshape((3,200,48))

f, ax = plt.subplots(3,2,figsize=(8,4))
for i, img in enumerate([wd_vis, cv_vis]):
    for j in range(3):
        ax[j][i].imshow(img[j].transpose(), cmap='nipy_spectral') 
        # ax[j][i].imshow(normalize(img[j].transpose()), cmap='nipy_spectral') 
        ax[j][i].set_xticks([])
        ax[j][i].set_yticks(np.arange(3, 48, 8))
        ax[j][i].set_yticklabels(vis_chns)
    ax[2][i].set_xticks(np.arange(0,220,40))  
    ax[2][i].set_xticklabels(['{:.01f}'.format(t) for t in np.arange(0,1.2,0.2)])
    ax[2][i].set_xlabel('t (s)')
plt.subplots_adjust(wspace=0.0, hspace=0.00)
plt.tight_layout()

f, ax = plt.subplots(3,2)
for j in range(3):
    ax[j][0].imshow(wd_f[j,...,-1].transpose(), cmap='nipy_spectral') 
    ax[j][0].set_xticks([])
    ax[j][0].set_yticks(np.arange(0,62, 10))
    # ax[j][0].set_yticklabels(vis_chns)
    
    ax[j][1].imshow(cv_f[j,...,0].transpose(), cmap='nipy_spectral') 
    ax[j][1].set_xticks([])
    ax[j][1].set_yticks(np.arange(0,62, 10))
    # ax[j][1].set_yticklabels(vis_chns)
    
ax[2][0].set_xticks(np.arange(0,220,40))  
ax[2][0].set_xticklabels(['{:.01f}'.format(t) for t in np.arange(0,1.2,0.2)])
ax[2][0].set_xlabel('t (s)')
ax[2][1].set_xticks(np.arange(0,220,40))  
ax[2][1].set_xticklabels(['{:.01f}'.format(t) for t in np.arange(0,1.2,0.2)])
ax[2][1].set_xlabel('t (s)')
plt.tight_layout()

# from pyevtk.hl import gridToVTK
# def save_to_vtk(data, filepath):
#     """
#     save the 3d data to a .vtk file. 
    
#     Parameters
#     ------------
#     data : 3d np.array
#         3d matrix that we want to visualize
#     filepath : str
#         where to save the vtk model, do not include vtk extension, it does automatically
#     """
#     x = np.arange(data.shape[0]+1)
#     y = np.arange(data.shape[1]+1)
#     z = np.arange(data.shape[2]+1)
#     gridToVTK(filepath, x, y, z, cellData={'data':data.copy()})
    
# save_to_vtk(wd_f[0], r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\wdf_0')
# save_to_vtk(wd_f[1], r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\wdf_1')
# save_to_vtk(wd_f[2], r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\wdf_2')
# save_to_vtk(cv_f[0], r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\cvf_0')
# save_to_vtk(cv_f[1], r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\cvf_1')
# save_to_vtk(cv_f[2], r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\cvf_2')

#%%
wd62 = np.load( os.path.join(st_path, 'benchmark_summary', 'CM_S{:02d}_proposed_2dwd_62chns.npy'.format(subject_for_wd)) )
# wd62 = np.load(r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\CM_S{}_proposed_2dwd_62chns.npy'.format(subject))
# wd12 = np.load(r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\CM_S{}_proposed_2dwd_12chns.npy'.format(subject))

cv62 = np.load( os.path.join(st_path, 'benchmark_summary', 'CM_S{:02d}_proposed_2d_62chns.npy'.format(subject_for_wd)) )
# cv62 = np.load(r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\CM_S{}_proposed_2d_62chns.npy'.format(subject))
# cv12 = np.load(r'C:\Users\dykua\github\EEG-decoding\benchmark_summary\CM_S{}_proposed_2d_12chns.npy'.format(subject))
               

plot_confusion_matrix(np.sum(wd62, axis=0), ['Negative', 'Neutral', 'Positive'], True)
plt.title('Wavelet Decomposition')
plot_confusion_matrix(np.sum(cv62,axis=0), ['Negative', 'Neutral', 'Positive'], True)
plt.title('Conv2d')

'''Averaging all'''
wd = []
cv = []
for subject_for_wd in range(1,16):
    wd62 = np.load( os.path.join(st_path, 'benchmark_summary', 'CM_S{:02d}_proposed_2dwd_62chns.npy'.format(subject_for_wd)) )

    cv62 = np.load( os.path.join(st_path, 'benchmark_summary', 'CM_S{:02d}_proposed_2d_62chns.npy'.format(subject_for_wd)) )

    wd.append(np.sum(wd62, axis=0)) 
    cv.append(np.sum(cv62, axis=0))   
    
plot_confusion_matrix(np.sum(np.array(wd), axis=0), ['Negative', 'Neutral', 'Positive'], True)
plt.title('Wavelet Decomposition')
plot_confusion_matrix(np.sum(np.array(cv),axis=0), ['Negative', 'Neutral', 'Positive'], True)
plt.title('Conv2d')

# %%
'''
Making the change of accuracy when inputs are band stoped at different frequencies.
two types:
  Same network: detailed to the 3 catgories
  different networks: overall change of accuracy.
'''
'''
bandpass filter, copied from 
https://users.soe.ucsc.edu/~karplus/bme51/w17/bandpass-filter.py
''' 
# from scipy import signal
# def batch_band_pass(values, low_end_cutoff, high_end_cutoff, sampling_freq, btype='bandpass'):
#     assert len(values.shape) == 3, "wrong input shape"
#     S, T, C = values.shape
#     X_filtered = np.empty(values.shape)
#     lo_end_over_Nyquist = low_end_cutoff/(0.5*sampling_freq)
#     hi_end_over_Nyquist = high_end_cutoff/(0.5*sampling_freq)

#     bess_b,bess_a = signal.iirfilter(5,
#                 Wn=[lo_end_over_Nyquist,hi_end_over_Nyquist],
#                 btype=btype, ftype='bessel')
                
#     for i in range(S):
#         for j in range(C):
#             X_filtered[i,:,j] = signal.filtfilt(bess_b,bess_a,values[i,:,j])
    
#     return X_filtered

# rej_band = [(l, l+10) for l in range(10,90,10)]
# rej_band.append((90, 99.99))
# rej_band.insert(0, (1,10))

# freq_p = []
# acc_list = []
# for subject in range(1,16):
#     X = loadmat( os.path.join(path, 'S{:02d}_E01.mat'.format(subject)) )['segs'].transpose([2,1,0])
#     Y = loadmat( os.path.join(path, 'Label.mat') )['seg_labels'][0]
#     data_seq = [ batch_band_pass(X, lp, hp, 200, btype='bandstop') for (lp,hp) in rej_band ] #bandstop case
#     data_seq.append(X)
#     data_seq = [zscore(_d, axis=1) for _d in data_seq]
    

#     CM_sum = [0]*len(data_seq)
#     for fld in range(5):
#         cpt_path = os.path.join( st_path, 'tmp', 'S{:02d}_checkpoint_proposed_2d_{}chns_fold{}'.format(subject, chns_token, fld) )
#         model.load_weights(cpt_path)

#         for ind, _data in enumerate(data_seq):
#             pred = model.predict(_data)
#             CM = confusion_matrix( Y+1, np.argmax(pred, axis=1) )
#             CM_sum[ind] += CM

#     freq_p.append(CM_sum)
#     acc_list.append( [[c[i,i]/np.sum(c[i]) for i in range(3)] for c in CM_sum] )
    
# acc_array = np.array(acc_list)
# # %%

# plt.figure()
# freq_grid = [10*i for i in range(1,11)]
# ll = ['Negative', 'Neutral',  'Positive']
# for _c in range(3):
#     plt.plot(acc_array[0,:,_c], 'd--',label = ll[_c])
# plt.ylim([0.5, 1.0])
# plt.xticks(np.arange(len(CM_sum)), 
#         labels=['1-10', '10-20', '20-30', '30-40', '40-50', '50-60',
#                 '60-70', '70-80', '80-90', '90-99.99', 'original'], rotation=-45)
# plt.legend(loc = 'lower left')
# plt.xlabel('Hz')
# plt.ylabel('Acc')
# plt.grid()



# %%
acc_array = np.load(r'E:\Benchmarks\SEED\benchmark_summary\bandstop_summary.npy')
freq_grid = [10*i for i in range(1,12)]
ll = ['Negative', 'Neutral',  'Positive']
cate_num = np.array([np.sum(Y==i)/len(Y) for i in range(-1,2)])
fig, ax = plt.subplots(3,5)
for i in range(15):
    r = i//5
    c = i%5
    for _c in range(3):
        ax[r][c].plot(acc_array[i,:,_c], 'd--',label = ll[_c])
    ax[r][c].plot(acc_array[i]@cate_num, label='Overall')

    ax[r][c].grid(True)
    ax[r][c].set_title('S{:02d}'.format(i+1))
    # ax[r][c].text(0.2, 0.2, 'S{:02d}'.format(i+1))
    ax[r][c].set_xticks(np.arange(len(freq_grid)) )
    ax[r][c].set_ylim(0.5, 1.0) 

    if c==0:
        ax[r][c].set_ylabel('Acc.')
  
    if r == 2:         
        ax[r][c].set_xticklabels(
                            labels=['1-10', '10-20', '20-30', '30-40', '40-50', '50-60',
                                    '60-70', '70-80', '80-90', '90-99.99', 'original'], rotation=-45)
        ax[r][c].set_xlabel('Hz')
        if c == 4:
            ax[r][c].legend()
    else:
        ax[r][c].set_xticklabels(labels=[])
        
        

    
# %%
'''Ablation for single branches'''
both = []
S = []
C = []
for _s in range(1,16): # only take the F1-score
    both.append( np.mean(np.load( os.path.join(st_path, 'benchmark_summary', 'S{:02d}_proposed_2d_62chns.npy'.format(_s)))[:,-1], axis=0) )
    S.append( np.mean(np.load( os.path.join(st_path, 'benchmark_summary', 'S{:02d}_proposed_2d_S_62chns.npy'.format(_s)))[:,-1], axis=0) )
    C.append( np.mean(np.load( os.path.join(st_path, 'benchmark_summary', 'S{:02d}_proposed_2d_C_62chns.npy'.format(_s)))[:,-1], axis=0) )

#%%
plt.figure()    
plt.plot(np.arange(1,16), both, '^--',label='Proposed', markersize=7, linewidth=2.0)
plt.plot(np.arange(1,16), S, '^--', label = 'Scale stream only', markersize=7)
plt.plot(np.arange(1,16), C, '^--',label='Channel stream only', markersize=7)
plt.legend()
plt.xlabel('Subject')
plt.ylabel('F1')
plt.grid(axis='y')
# %%
