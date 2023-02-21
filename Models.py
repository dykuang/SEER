# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:49:31 2020

@author: dykua
"""

from tensorflow.keras.layers import Layer, Input, DepthwiseConv2D, SeparableConv1D, Conv1D, Lambda, Multiply, Add, Dense, Reshape, BatchNormalization, Activation, Dropout, Permute, Flatten, concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
import numpy as np
from tensorflow.keras import losses
from tensorflow.keras.layers import GaussianDropout, SpatialDropout2D, SpatialDropout1D, Concatenate, MaxPooling1D
from tensorflow.keras.regularizers import l1, l2
import tensorflow as tf
from Modules import damped_log_loss

#%%
class Weight(Layer):
    def __init__(self):
        super(Weight, self).__init__()
        
    def build(self, input_shape):
        assert isinstance(input_shape, list), 'Inputs should be a list of tensors.'
        self.w = self.add_weight(
            shape=(1,),
            initializer="random_uniform",
            trainable=True,
        )
        # self.b = self.add_weight(
        #     shape=(self.units,), initializer="random_normal", trainable=True
        # )

    def call(self, inputs):
        assert isinstance(inputs, list), 'Inputs should be a list of tensors.'
        return self.w*inputs[0] + (1 - self.w)*inputs[1]
    
class C_Weight(Layer):
    def __init__(self):
        super(C_Weight, self).__init__()
        
    def build(self, input_shape):
        assert isinstance(input_shape, list), 'Inputs should be a list of tensors.'
        c = input_shape[0][-1]
        self.w = self.add_weight(
            shape=(1,1,c),
            initializer="random_uniform",
            trainable=True,
        )
        # self.b = self.add_weight(
        #     shape=(self.units,), initializer="random_normal", trainable=True
        # )

    def call(self, inputs):
        assert isinstance(inputs, list), 'Inputs should be a list of tensors.'
        return self.w*inputs[0] + (1 - self.w)*inputs[1]

'''
Discriminative models
'''


'''
Second approach modified from    
http://asymptoticlabs.com/blog/posts/waveletSpectrogramsTFSR.html
'''
#%%
from tensorflow.compat.v1.keras import initializers
from tensorflow.keras.layers import AveragePooling1D, UpSampling1D


def press_tensor(input_tensor, 
                 target_length, 
                 min_pool_size,
                 pooling_type="average"):
    """Resize a tensor to a desired shape by upsampling or averaging (or some combination)"""
    cur_size = int(input_tensor.shape[1])
    if cur_size < target_length:
        input_tensor = UpSampling1D(target_length//cur_size)(input_tensor)
    size_ratio = max(1, cur_size//target_length)
    pool_size = max(min_pool_size, size_ratio)
    if not (pool_size == 1 and size_ratio == 1):
        return AveragePooling1D(pool_size=pool_size, strides=size_ratio, padding="same")(input_tensor)
    else:
        return input_tensor
    
    
def make_wavelet_expansion(
    input_tensor,
    num_filters,
    strides,
    n_levels,
    low_pass_filter,
    high_pass_filter,
    trainable=False,
    channel = None,
):
    wv_kwargs = {
        "filters":num_filters, 
        "kernel_size":len(low_pass_filter),
        "strides":strides,     
        "use_bias":False, 
        "padding":"same", 
        "trainable":trainable,
    }
    
    approximation_coefficients = []
    detail_coefficients = []
    
    last_approximant = input_tensor
    # print(last_approximant.shape)
    for i in range(n_levels):
        lpf = low_pass_filter
        hpf = high_pass_filter
        a_n = Conv1D(
            kernel_initializer=initializers.Constant(lpf.reshape((-1, 1))),
            name="low_pass_{}_{}".format(i, channel),
            **wv_kwargs
        )(last_approximant)
        d_n = Conv1D(
            kernel_initializer=initializers.Constant(hpf.reshape((-1, 1))),
            name="high_pass_{}_{}".format(i, channel),
            **wv_kwargs,
        )(last_approximant)
        
        detail_coefficients.append(d_n)
        approximation_coefficients.append(a_n)
        last_approximant = a_n
    
    return approximation_coefficients, detail_coefficients

#%%
def wavelet_net(in_shape,
                 low_pass, high_pass,
                 num_filters=1, strides = 2,
                 target_size=128, min_pool_size =3, 
                 n_levels = 3,
                 wavelets_trainable = True):
    
    signal_in = Input(shape = in_shape)
    output = []
    for i in range(signal_in.shape[-1]):
        approx_stack, detail_stack = make_wavelet_expansion(
                                                            signal_in[...,i][...,None], 
                                                            num_filters=num_filters,
                                                            strides = strides,
                                                            n_levels = n_levels, 
                                                            low_pass_filter=low_pass, 
                                                            high_pass_filter=high_pass,
                                                            trainable=wavelets_trainable,
                                                            channel = i
                                                            )
        
        features_list = []
        features_list.extend(detail_stack)
        features_list.append(approx_stack[-1])
        
        resized_features = []
    
        for wv_in in features_list:
            x = BatchNormalization()(wv_in)#normalize each filter before taking the power
            x = Lambda(lambda x: tf.square(x))(x)#square the coefficients to get a power estimate
            resized_features.append(press_tensor(x, target_size, min_pool_size))
    
        #concatenate the features together to get our spectrogram
        specgram = concatenate(resized_features, axis=-1)
        # print(specgram.shape)
        log_specgram = Lambda(lambda x: tf.math.log(1.0+x))(specgram)
        output.append(log_specgram)
    
    stack = tf.concat([out[...,None,:] for out in output], axis=2)
    
    
    return Model(signal_in, stack )

#%%
def TFCNet_WM(WM_model, 
              num_classes, 
              dep_activation = 'tanh', sep_activation = 'linear',
              depth_multiplier = 8, depth_rate=2,
              kernel_num=32, kernel_len=5,
              num_filters_list = [32, 64], kernel_size_list=[5,5],
              strides_for_pool=[5,5],
              optimizer=Adam, learning_rate=1e-3, droprate=0.5, normrate_head=1.0, normrate_dense = 0.5):
    
    in_shape = WM_model.input_shape[1:]
    out_shape = WM_model.output_shape[2:]
    x_in = Input(in_shape)
    x = WM_model(x_in)
    
    x = BatchNormalization(axis=1)(x)
    x = Dropout(droprate)(x)
     
    x = fork_merge(x, out_shape,  
                    depth_rate,
                    depth_multiplier, 
                    kernel_num,  kernel_len, 
                    dropout_rate= droprate, 
                    normrate_head = normrate_head,
                    dep_activation = dep_activation,
                    sep_activation = sep_activation)
    x = BatchNormalization(axis=-1)(x)
    x = GaussianDropout(0.2)(x)
    
    for i in range(len(strides_for_pool)):
        x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], 
                            # depthwise_constraint=max_norm(normrate_head/2),
                            # activity_regularizer=l1(1e-5),
                            activation=None,use_bias = True, name='sepconv-{}'.format(i))(x)
        # x = BatchNormalization(axis=-1)(x)
        x = Activation('elu')(x)
        x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], strides = strides_for_pool[i],
                            # depthwise_constraint=max_norm(normrate_head/2),
                            # activity_regularizer=l1(1e-5),
                            activation=None, use_bias = True, name='pooling-{}'.format(i))(x)
        x = BatchNormalization(axis=-1)(x)
        x = SpatialDropout1D(0.2)(x)
        x = Activation('elu')(x)


    
    x = Dropout(droprate)(x)
    
    x = Flatten(name = 'flatten')(x)
    x = Dense(num_classes, name = 'dense', kernel_constraint = max_norm(normrate_dense) )(x)
    
    x_out = Activation('softmax', name = 'softmax')(x)
    
    Mymodel = Model(x_in, x_out)
    Mymodel.compile(
                    loss='categorical_crossentropy', 
                    # loss = damped_log_loss,
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    
    return Mymodel

#%%
def fork_merge(x_in, shape, 
               depth_rate,
               depth_multiplier, 
               kernel_num, kernel_len, dropout_rate=0.2, 
               normrate_head = 0.5,
               dep_activation='tanh',
               sep_activation = 'linear',
               merge_style = 'A',
               _label = None):
    
    #The F branch ==================================
    x = DepthwiseConv2D((1, shape[0]), strides=(1, 1), padding="valid",
                    depth_multiplier=depth_multiplier,
                    data_format=None, dilation_rate=(1, 1), 
                    activation=None,  use_bias=False,
                    depthwise_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    depthwise_regularizer=None, bias_regularizer=None,
                    activity_regularizer=None, depthwise_constraint=max_norm(normrate_head),
                    bias_constraint=None, name = 'F_Dep_{}'.format(_label) )(x_in)
    x = BatchNormalization(momentum=0.9, axis=-1)(x)
    x = SpatialDropout2D(dropout_rate)(x)
    x = Activation(dep_activation)(x)
    
    # x = Permute((-2,-1))(x)
    # x = Reshape((-1, in_shape[-1]))(x)
    x = Lambda(lambda y: y[:, :, 0, :])(x)
    
    x = SeparableConv1D( kernel_num, kernel_size=kernel_len, strides=1, padding="same",
                         data_format=None, dilation_rate=1, depth_multiplier= 1,
                         activation=sep_activation, use_bias=False, depthwise_initializer="glorot_normal",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None, name = 'F_Sep_{}'.format(_label)
                         )(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Dropout(droprate)(x)
    
    
    # The C branch =======================================================
    xt = Permute((1,3,2))(x_in)
    # xt = GaussianDropout(0.5)(xt)
    xt = DepthwiseConv2D((1, shape[-1]), strides=(1, 1), padding="valid",
                         depth_multiplier=depth_multiplier,
                         activation=None,  use_bias=False,
                         depthwise_initializer="glorot_uniform",
                         bias_initializer="zeros",
                         depthwise_constraint=max_norm(normrate_head),
                         name = 'C_Dep_{}'.format(_label))(xt)
    xt = BatchNormalization(momentum=0.9,axis=-1)(xt)
    xt = SpatialDropout2D(dropout_rate)(xt)
    xt = Activation(dep_activation)(xt)
    xt = Lambda(lambda y: y[:, :, 0, :])(xt)
    
    xt = SeparableConv1D( kernel_num, kernel_size=kernel_len, strides=1, padding="same",
                         data_format=None, dilation_rate=1, depth_multiplier=1,
                         activation=sep_activation, use_bias=False, depthwise_initializer="glorot_normal",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None, name = 'C_Sep_{}'.format(_label)
                         )(xt)
    
    # xt = Dropout(droprate)(xt)
    # Merge two branches =====================================================
    
    if merge_style == 'A':
        # x = attach_attention_module(x, 'se_block', ratio=2)   
        x = Add()([x, xt])  

    elif merge_style == 'C':
        x = Concatenate()([x, xt])
    elif merge_style == 'M':
        x = Multiply()([x, xt])
    elif merge_style == 'W':
        x = Weight()([x, xt])
    elif merge_style == 'CW':
        x = C_Weight()([x, xt])
    
    return x

#%%
from Modules import WaveletDeconvolution, attach_attention_module

def TFCNet_multiWD(in_shape, num_classes, 
                   dep_activation = 'tanh', sep_activation = 'linear',
                   merge_style = 'A', use_WD = False,
                   WDspec_list = [[8, 5, 1]], # Number, len, strides
                   depth_multiplier = 1, depth_rate=1, #WD_channels = 16,
                   merge_kernel_num = 8, merge_kernel_len = 5,
                   num_filters_list = [16, 32], kernel_size_list=[5,5],
                   strides_for_pool=[5,5],
                   optimizer=Adam, learning_rate=1e-3, 
                   droprate=0.5, spatial_droprate=0.2,
                   normrate_head=1.0, normrate_dense = 0.5):
    
    x_in = Input(shape = in_shape, name = 'input')
    # x_wd = GaussianDropout(0.1)(x_in)
    # x_wd = Dropout(0.5)(x_in)
    # x_wd = WaveletDeconvolution(WD_spec[0], kernel_length=WD_spec[1], strides=WD_spec[2], use_bias = True,
    #                           padding='same', data_format='channels_last', name='WD-1')(x_wd)
    # x_wd = BatchNormalization(axis=1)(x_in)
    if use_WD:
    
        x_wd = WaveletDeconvolution(WDspec_list[0][0], kernel_length=WDspec_list[0][1], strides=WDspec_list[0][2], 
                                    use_bias = False,
                                    padding='same', data_format='channels_last', name='WD-{}'.format(0))(x_in)
    # x_wd_1 = WaveletDeconvolution(WDspec_list[0][0], kernel_length=2*WDspec_list[0][1], strides=WDspec_list[0][2], 
    #                               use_bias = False,
    #                               padding='same', data_format='channels_last', name='WD-{}-1'.format(0))(x_in)
    # x_wd_2 = WaveletDeconvolution(WDspec_list[0][0], kernel_length=4*WDspec_list[0][1], strides=WDspec_list[0][2], 
    #                               use_bias = False,
    #                               padding='same', data_format='channels_last', name='WD-{}-2'.format(0))(x_in)
    
    # x_wd = concatenate([x_wd, x_wd_1, x_wd_2], axis=-2)
    
    # # x_wd = BatchNormalization(axis=1)(x_wd) # which dimension to normalize?
    # x_wd = SpatialDropout2D(spatial_droprate)(x_wd)
    else:
        x_wd = Lambda(lambda x: x[...,None])(x_in)
        x_wd = Conv2D(WDspec_list[0][0], kernel_size=(WDspec_list[0][1],1), strides=WDspec_list[0][2], 
                      kernel_initializer = 'glorot_normal',
                      # groups = 1,
                      kernel_constraint=max_norm(normrate_head),
                      use_bias = False,
                      padding='same', data_format='channels_last', name='WD-{}'.format(0))(x_wd)
        x_wd = BatchNormalization(momentum=0.9, axis=-1)(x_wd)
        x_wd = Permute((1,3,2))(x_wd)
   
    
    x = fork_merge(x_wd, (WDspec_list[0][0], in_shape[-1]), 
                   depth_rate,
                   depth_multiplier, 
                   merge_kernel_num, merge_kernel_len, dropout_rate=spatial_droprate, 
                   normrate_head = normrate_head,
                   dep_activation = dep_activation,
                   sep_activation = sep_activation,
                   merge_style = merge_style,
                   _label = 0)
    x = BatchNormalization(momentum=0.9,axis=-1)(x)
    # x = Activation('elu')(x)
    # x = GaussianDropout(droprate)(x)
    x = SpatialDropout1D(spatial_droprate)(x)
    
    for i, spec in enumerate(WDspec_list[1:]):
        if use_WD:
            x = WaveletDeconvolution(spec[0], kernel_length=spec[1], strides=spec[2], use_bias = False,
                                      padding='same', data_format='channels_last', name='WD-{}'.format(i+1))(x)
        else:
            x = Lambda(lambda x: x[...,None])(x)
            x = Conv2D(spec[0], kernel_size=(spec[1], 1) , strides=spec[2], use_bias = False,
                       kernel_initializer = 'glorot_normal',
                       # groups = 1,
                       kernel_constraint=max_norm(normrate_head),
                       padding='same', data_format='channels_last', name='WD-{}'.format(i+1))(x)
            x = BatchNormalization(momentum=0.9,axis=-1)(x)
            x = Permute((1,3,2))(x)
            
        x = SpatialDropout2D(spatial_droprate)(x)  
        
        x = fork_merge(x, (spec[0], merge_kernel_num), 
                       depth_rate,
                       depth_multiplier, 
                       merge_kernel_num, merge_kernel_len, 
                       dropout_rate=spatial_droprate, 
                       normrate_head = normrate_head,
                       dep_activation=dep_activation,
                       sep_activation = sep_activation,
                       _label = i+1)
        x = BatchNormalization(momentum=0.9, axis=-1)(x)
        # x = Activation('elu')(x)
        # x = GaussianDropout(spatial_droprate)(x)
        # x = SpatialDropout1D(spatial_droprate)(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = GaussianDropout(spatial_droprate)(x)
    x = SpatialDropout1D(spatial_droprate)(x)
    for i in range(len(strides_for_pool)):
        # x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], padding='same',
        #                     # depthwise_constraint=max_norm(normrate_head/2),
        #                     # activity_regularizer=l1(1e-5),
        #                     activation=None, use_bias = False, name='sepconv-{}'.format(i))(x)
        # # x = BatchNormalization(axis=-1)(x)
        # x = Activation('elu')(x)
        # x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], strides = strides_for_pool[i],
        #                     # depthwise_constraint=max_norm(normrate_head/2),
        #                     # activity_regularizer=l1(1e-5),
        #                     padding='same',
        #                     activation=None, use_bias = False, name='pooling-{}'.format(i))(x)
        # x = BatchNormalization(axis=-1)(x)
        # x = SpatialDropout1D(spatial_droprate)(x)
        # x = Activation('elu')(x)
        
        
        # try pooling
        x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], strides = 1,
                            # depthwise_constraint=max_norm(normrate_head/2),
                            # activity_regularizer=l1(1e-5),
                            depthwise_initializer="glorot_normal",
                            pointwise_initializer="glorot_uniform",
                            activation=None, use_bias = False)(x)
        x = BatchNormalization(momentum=0.9, axis=-1)(x)
        x = SpatialDropout1D(spatial_droprate)(x)
        x = Activation('elu')(x)
        x = MaxPooling1D(strides_for_pool[i], name='pooling-{}'.format(i))(x)

    # x = Lambda(lambda x: x[...,None,:])(x)
    # x = attach_attention_module(x, 'se_block', ratio=2)    
    # x = Lambda(lambda x: x[...,0,:])(x)

    # x = SeparableConv1D(num_filters_list[2], kernel_size=1, use_bias = True, name='sepconv-3')(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Activation('elu')(x)
    
    x = Dropout(droprate)(x)
    
    x = Flatten(name = 'flatten')(x)
    # x = GlobalAveragePooling1D(name = 'flatten')(x)
    
    # x = Dense(32, name='feature', activation = 'elu')(x)
    
    x = Dense(num_classes, name = 'dense_last', kernel_constraint = max_norm(normrate_dense) )(x)
    
    softmax = Activation('softmax', name = 'softmax')(x)
    
    Mymodel = Model(x_in, softmax)
    Mymodel.compile(
                    loss='categorical_crossentropy', 
                    # loss = damped_log_loss,
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    
    return Mymodel


#%%
def fork_merge_2d(x_in, shape, 
                   depth_rate,
                   depth_multiplier, 
                   kernel_num, kernel_len, dropout_rate=0.2, 
                   normrate_head = 0.5,
                   dep_activation='tanh',
                   sep_activation = 'linear',
                   merge_style = 'A',
                   _label = None):
    
    #The F branch ==================================
    x = DepthwiseConv2D((1, shape[0]), strides=(1, 1), padding="valid",
                    depth_multiplier=depth_multiplier,
                    data_format=None, dilation_rate=(1, 1), 
                    activation=None,  use_bias=False,
                    depthwise_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    depthwise_regularizer=None, bias_regularizer=None,
                    activity_regularizer=None, depthwise_constraint=max_norm(normrate_head),
                    bias_constraint=None, name = 'F_Dep_{}'.format(_label) )(x_in)
    x = BatchNormalization(axis=-1)(x)
    x = SpatialDropout2D(dropout_rate)(x)
    x = Activation(dep_activation)(x)
    
    
    x = SeparableConv2D( kernel_num, kernel_size=(kernel_len,1), strides=1, padding="same",
                         data_format=None, dilation_rate=1, depth_multiplier= 1,
                         activation=sep_activation, use_bias=False, depthwise_initializer="glorot_normal",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None, name = 'F_Sep_{}'.format(_label)
                         )(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Dropout(droprate)(x)
    
    
    # The C branch =======================================================
    xt = Permute((1,3,2))(x_in)
    # xt = GaussianDropout(0.5)(xt)
    xt = DepthwiseConv2D((1, shape[-1]), strides=(1, 1), padding="valid",
                         depth_multiplier=depth_multiplier,
                         activation=None,  use_bias=False,
                         depthwise_initializer="glorot_uniform",
                         bias_initializer="zeros",
                         depthwise_constraint=max_norm(normrate_head),
                         name = 'C_Dep_{}'.format(_label))(xt)
    xt = BatchNormalization(axis=-1)(xt)
    xt = SpatialDropout2D(dropout_rate)(xt)
    xt = Activation(dep_activation)(xt)
    
    xt = SeparableConv2D( kernel_num, kernel_size=(kernel_len,1), strides=1, padding="same",
                         data_format=None, dilation_rate=1, depth_multiplier=1,
                         activation=sep_activation, use_bias=False, depthwise_initializer="glorot_normal",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None, name = 'C_Sep_{}'.format(_label)
                         )(xt)
    
    # xt = Dropout(droprate)(xt)
    # Merge two branches =====================================================
    
    if merge_style == 'A':
        x = Add()([x, xt])   
    elif merge_style == 'C':
        x = Concatenate()([x, xt])
    elif merge_style == 'M':
        x = Multiply()([x, xt])
    elif merge_style == 'W':
        x = Weight()([x, xt])
    elif merge_style == 'CW':
        x = C_Weight()([x, xt])
    
    return x

def fork_merge_2d_abl(x_in, shape, 
                   depth_rate,
                   depth_multiplier, 
                   kernel_num, kernel_len, dropout_rate=0.2, 
                   normrate_head = 0.5,
                   dep_activation='tanh',
                   sep_activation = 'linear',
                   merge_style = 'A',
                   mode = 'S',
                   _label = None):
    '''
    another version of fork merge_2d for the ablation study
    mode:
        if 'merge', same as fork_merge_2d
        if 'S': only returns the F branch
        if 'C': only returns the C branch 
    '''
    
    if mode == 'S':
        #The F branch ==================================
        x = DepthwiseConv2D((1, shape[0]), strides=(1, 1), padding="valid",
                        depth_multiplier=depth_multiplier,
                        data_format=None, dilation_rate=(1, 1), 
                        activation=None,  use_bias=False,
                        depthwise_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        depthwise_regularizer=None, bias_regularizer=None,
                        activity_regularizer=None, depthwise_constraint=max_norm(normrate_head),
                        bias_constraint=None, name = 'F_Dep_{}'.format(_label) )(x_in)
        x = BatchNormalization(axis=-1)(x)
        x = SpatialDropout2D(dropout_rate)(x)
        x = Activation(dep_activation)(x)
        
        
        x = SeparableConv2D( kernel_num, kernel_size=(kernel_len,1), strides=1, padding="same",
                            data_format=None, dilation_rate=1, depth_multiplier= 1,
                            activation=sep_activation, use_bias=False, depthwise_initializer="glorot_normal",
                            pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                            depthwise_regularizer=None, pointwise_regularizer=None,
                            bias_regularizer=None,  activity_regularizer=None,
                            depthwise_constraint=None, pointwise_constraint=None,
                            bias_constraint=None, name = 'F_Sep_{}'.format(_label)
                            )(x)
        # x = BatchNormalization(axis=1)(x)
        # x = Dropout(droprate)(x)

        return x
    
    elif mode == 'C':
        # The C branch =======================================================
        xt = Permute((1,3,2))(x_in)
        # xt = GaussianDropout(0.5)(xt)
        xt = DepthwiseConv2D((1, shape[-1]), strides=(1, 1), padding="valid",
                            depth_multiplier=depth_multiplier,
                            activation=None,  use_bias=False,
                            depthwise_initializer="glorot_uniform",
                            bias_initializer="zeros",
                            depthwise_constraint=max_norm(normrate_head),
                            name = 'C_Dep_{}'.format(_label))(xt)
        xt = BatchNormalization(axis=-1)(xt)
        xt = SpatialDropout2D(dropout_rate)(xt)
        xt = Activation(dep_activation)(xt)
        
        xt = SeparableConv2D( kernel_num, kernel_size=(kernel_len,1), strides=1, padding="same",
                            data_format=None, dilation_rate=1, depth_multiplier=1,
                            activation=sep_activation, use_bias=False, depthwise_initializer="glorot_normal",
                            pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                            depthwise_regularizer=None, pointwise_regularizer=None,
                            bias_regularizer=None,  activity_regularizer=None,
                            depthwise_constraint=None, pointwise_constraint=None,
                            bias_constraint=None, name = 'C_Sep_{}'.format(_label)
                            )(xt)
        
        # xt = Dropout(droprate)(xt)
        return xt
    
    elif mode =='Merge':
    # Merge two branches =====================================================
        return fork_merge_2d(x_in, shape, 
                            depth_rate,
                            depth_multiplier, 
                            kernel_num, kernel_len, dropout_rate, 
                            normrate_head ,
                            dep_activation,
                            sep_activation,
                            merge_style ,
                            _label)
    else:
        raise NotImplementedError

def TFCNet_multiWD_2d(in_shape, num_classes, 
                       dep_activation = 'tanh', sep_activation = 'linear',
                       merge_style = 'A', use_WD = False,
                       WDspec_list = [[8, 5, 1]], # Number, len, strides
                       depth_multiplier = 1, depth_rate=1, #WD_channels = 16,
                       merge_kernel_num = 8, merge_kernel_len = 5,
                       num_filters_list = [16, 32], kernel_size_list=[5,5],
                       strides_for_pool=[5,5],
                       optimizer=Adam, learning_rate=1e-3, 
                       droprate=0.5, spatial_droprate=0.2,
                       normrate_head=1.0, normrate_dense = 0.5):
    
    x_in = Input(shape = in_shape, name = 'input')
    # x_wd = Lambda(lambda x: x[...,None])(x_in)
    # x_wd = GaussianDropout(0.1)(x_in)
    # x_wd = Dropout(0.5)(x_in)
    # x_wd = WaveletDeconvolution(WD_spec[0], kernel_length=WD_spec[1], strides=WD_spec[2], use_bias = True,
    #                           padding='same', data_format='channels_last', name='WD-1')(x_wd)
    # x_wd = BatchNormalization(axis=1)(x_in)
    if use_WD:
        x_wd = Lambda(lambda x: x[..., 0])(x_in)
        x_wd = WaveletDeconvolution(WDspec_list[0][0], kernel_length=WDspec_list[0][1], strides=WDspec_list[0][2], 
                                    use_bias = False,
                                    padding='same', data_format='channels_last', name='WD-{}'.format(0))(x_wd)

    else:
        # x_wd = Lambda(lambda x: x[...,None])(x_in)
        x_wd = Conv2D(WDspec_list[0][0], kernel_size=(WDspec_list[0][1],1), strides=WDspec_list[0][2], 
                      kernel_initializer = 'glorot_normal',
                      # groups = 1,
                      kernel_constraint=max_norm(normrate_head),
                      use_bias = False,
                      padding='same', data_format='channels_last', name='WD-{}'.format(0))(x_in)
        x_wd = BatchNormalization(axis=-1)(x_wd)
        x_wd = Permute((1,3,2))(x_wd)
   
    
    x = fork_merge_2d(x_wd, (WDspec_list[0][0], in_shape[1]), 
                       depth_rate,
                       depth_multiplier, 
                       merge_kernel_num, merge_kernel_len, dropout_rate=spatial_droprate, 
                       normrate_head = normrate_head,
                       dep_activation = dep_activation,
                       sep_activation = sep_activation,
                       merge_style = merge_style,
                       _label = 0)
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    x = GaussianDropout(droprate)(x)
    # x = SpatialDropout1D(spatial_droprate)(x)
    
    for i, spec in enumerate(WDspec_list[1:]):
        if use_WD:
            x = Lambda(lambda x: x[..., 0])(x)
            x = WaveletDeconvolution(spec[0], kernel_length=spec[1], strides=spec[2], use_bias = False,
                                      padding='same', data_format='channels_last', name='WD-{}'.format(i+1))(x)
        else:
            # x = Lambda(lambda x: x[...,None])(x)
            x = Conv2D(spec[0], kernel_size=(spec[1], 1) , strides=spec[2], use_bias = False,
                       kernel_initializer = 'glorot_normal',
                       # groups = 1,
                       kernel_constraint=max_norm(normrate_head),
                       padding='same', data_format='channels_last', name='WD-{}'.format(i+1))(x)
            x = BatchNormalization(axis=-1)(x)
            x = Permute((1,3,2))(x)
            
        x = SpatialDropout2D(spatial_droprate)(x)  
        
        x = fork_merge_2d(x, (spec[0], merge_kernel_num), 
                        depth_rate,
                        depth_multiplier, 
                        merge_kernel_num, merge_kernel_len, 
                        dropout_rate=spatial_droprate, 
                        normrate_head = normrate_head,
                        dep_activation=dep_activation,
                        sep_activation = sep_activation,
                        merge_style=merge_style,
                        _label = i+1)
        x = BatchNormalization(axis=-1)(x)
        # x = Activation('elu')(x)
        # x = GaussianDropout(spatial_droprate)(x)
        # x = SpatialDropout1D(spatial_droprate)(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = GaussianDropout(spatial_droprate)(x)
    x = SpatialDropout2D(spatial_droprate)(x)
    for i in range(len(strides_for_pool)):
        # x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], padding='same',
        #                     # depthwise_constraint=max_norm(normrate_head/2),
        #                     # activity_regularizer=l1(1e-5),
        #                     activation=None, use_bias = False, name='sepconv-{}'.format(i))(x)
        # # x = BatchNormalization(axis=-1)(x)
        # x = Activation('elu')(x)
        # x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], strides = strides_for_pool[i],
        #                     # depthwise_constraint=max_norm(normrate_head/2),
        #                     # activity_regularizer=l1(1e-5),
        #                     padding='same',
        #                     activation=None, use_bias = False, name='pooling-{}'.format(i))(x)
        # x = BatchNormalization(axis=-1)(x)
        # x = SpatialDropout1D(spatial_droprate)(x)
        # x = Activation('elu')(x)
        
        
        # try pooling
        x = SeparableConv2D(num_filters_list[i], kernel_size=(kernel_size_list[i],1), strides = 1,
                            # depthwise_constraint=max_norm(normrate_head/2),
                            # activity_regularizer=l1(1e-5),
                            depthwise_initializer="glorot_normal",
                            pointwise_initializer="glorot_uniform",
                            activation=None, use_bias = False)(x)
        x = BatchNormalization(axis=-1)(x)
        x = SpatialDropout2D(spatial_droprate)(x)
        x = Activation('elu')(x)
        x = MaxPooling2D( (strides_for_pool[i],1), name='pooling-{}'.format(i))(x)
        
    
    # x = SeparableConv1D(num_filters_list[2], kernel_size=1, use_bias = True, name='sepconv-3')(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Activation('elu')(x)
    
    x = Dropout(droprate)(x)
    
    x = Flatten(name = 'flatten')(x)
    # x = GlobalAveragePooling1D(name = 'flatten')(x)
    
    # x = Dense(32, name='feature', activation = 'elu')(x)
    
    x = Dense(num_classes, name = 'dense', kernel_constraint = max_norm(normrate_dense) )(x)
    
    softmax = Activation('softmax', name = 'softmax')(x)
    
    Mymodel = Model(x_in, softmax)
    Mymodel.compile(
                    loss='categorical_crossentropy', 
                    # loss = damped_log_loss,
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    
    return Mymodel

#%%
def TFCNet_multiWD_2d_abl(in_shape, num_classes, 
                       dep_activation = 'tanh', sep_activation = 'linear',
                       merge_style = 'A', mode='Merge', use_WD = False,
                       WDspec_list = [[8, 5, 1]], # Number, len, strides
                       depth_multiplier = 1, depth_rate=1, #WD_channels = 16,
                       merge_kernel_num = 8, merge_kernel_len = 5,
                       num_filters_list = [16, 32], kernel_size_list=[5,5],
                       strides_for_pool=[5,5],
                       optimizer=Adam, learning_rate=1e-3, 
                       droprate=0.5, spatial_droprate=0.2,
                       normrate_head=1.0, normrate_dense = 0.5,
                       ):
    '''
    For the ablation study where a single branch is used
    '''
    
    x_in = Input(shape = in_shape, name = 'input')
    # x_wd = Lambda(lambda x: x[...,None])(x_in)
    # x_wd = GaussianDropout(0.1)(x_in)
    # x_wd = Dropout(0.5)(x_in)
    # x_wd = WaveletDeconvolution(WD_spec[0], kernel_length=WD_spec[1], strides=WD_spec[2], use_bias = True,
    #                           padding='same', data_format='channels_last', name='WD-1')(x_wd)
    # x_wd = BatchNormalization(axis=1)(x_in)
    if use_WD:
        x_wd = Lambda(lambda x: x[..., 0])(x_in)
        x_wd = WaveletDeconvolution(WDspec_list[0][0], kernel_length=WDspec_list[0][1], strides=WDspec_list[0][2], 
                                    use_bias = False,
                                    padding='same', data_format='channels_last', name='WD-{}'.format(0))(x_wd)

    else:
        # x_wd = Lambda(lambda x: x[...,None])(x_in)
        x_wd = Conv2D(WDspec_list[0][0], kernel_size=(WDspec_list[0][1],1), strides=WDspec_list[0][2], 
                      kernel_initializer = 'glorot_normal',
                      # groups = 1,
                      kernel_constraint=max_norm(normrate_head),
                      use_bias = False,
                      padding='same', data_format='channels_last', name='WD-{}'.format(0))(x_in)
        x_wd = BatchNormalization(axis=-1)(x_wd)
        x_wd = Permute((1,3,2))(x_wd)
   
    
    x = fork_merge_2d_abl(x_wd, (WDspec_list[0][0], in_shape[1]), 
                       depth_rate,
                       depth_multiplier, 
                       merge_kernel_num, merge_kernel_len, dropout_rate=spatial_droprate, 
                       normrate_head = normrate_head,
                       dep_activation = dep_activation,
                       sep_activation = sep_activation,
                       mode=mode,merge_style = merge_style,
                       _label = 0)
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    x = GaussianDropout(droprate)(x)
    # x = SpatialDropout1D(spatial_droprate)(x)
    
    for i, spec in enumerate(WDspec_list[1:]):
        if use_WD:
            x = Lambda(lambda x: x[..., 0])(x)
            x = WaveletDeconvolution(spec[0], kernel_length=spec[1], strides=spec[2], use_bias = False,
                                      padding='same', data_format='channels_last', name='WD-{}'.format(i+1))(x)
        else:
            x = Conv2D(spec[0], kernel_size=(spec[1], 1) , strides=spec[2], use_bias = False,
                       kernel_initializer = 'glorot_normal',
                       # groups = 1,
                       kernel_constraint=max_norm(normrate_head),
                       padding='same', data_format='channels_last', name='WD-{}'.format(i+1))(x)
            x = BatchNormalization(axis=-1)(x)
            x = Permute((1,3,2))(x)
            
        x = SpatialDropout2D(spatial_droprate)(x)  
        
        x = fork_merge_2d_abl(x, (spec[0], merge_kernel_num), 
                        depth_rate,
                        depth_multiplier, 
                        merge_kernel_num, merge_kernel_len, 
                        dropout_rate=spatial_droprate, 
                        normrate_head = normrate_head,
                        dep_activation=dep_activation,
                        sep_activation = sep_activation,
                        mode=mode,merge_style = merge_style,
                        _label = i+1)
        x = BatchNormalization(axis=-1)(x)

    x = SpatialDropout2D(spatial_droprate)(x)
    for i in range(len(strides_for_pool)):   
        # try pooling
        x = SeparableConv2D(num_filters_list[i], kernel_size=(kernel_size_list[i],1), strides = 1,

                            depthwise_initializer="glorot_normal",
                            pointwise_initializer="glorot_uniform",
                            activation=None, use_bias = False)(x)
        x = BatchNormalization(axis=-1)(x)
        x = SpatialDropout2D(spatial_droprate)(x)
        x = Activation('elu')(x)
        x = MaxPooling2D( (strides_for_pool[i],1), name='pooling-{}'.format(i))(x)
        
    x = Dropout(droprate)(x)
    
    x = Flatten(name = 'flatten')(x)

    
    x = Dense(num_classes, name = 'dense', kernel_constraint = max_norm(normrate_dense) )(x)
    
    softmax = Activation('softmax', name = 'softmax')(x)
    
    Mymodel = Model(x_in, softmax)
    Mymodel.compile(
                    loss='categorical_crossentropy', 
                    # loss = damped_log_loss,
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    
    return Mymodel
    
#%%
def TFCNet_WD(in_shape, num_classes, 
              depth_activation = 'tanh', sep_activation = 'linear',
              WD_spec = [8, 5, 1], # Number, len, strides
              depth_multiplier = 8, depth_rate=2,
              num_filters_list = [16, 32, 64], kernel_size_list=[5,5,5],
              strides_for_pool=[5,5],
              optimizer=Adam, learning_rate=1e-3, droprate=0.5, normrate_head=1.0, normrate_dense = 0.5):
    '''
    in_shape = (time, channel, freqband)
    '''
    
    x_in = Input(shape = in_shape, name = 'input')
    # x_wd = GaussianDropout(0.5)(x_in)
    # x_wd = Dropout(0.5)(x_in)
    # x_wd = WaveletDeconvolution(WD_spec[0], kernel_length=WD_spec[1], strides=WD_spec[2], use_bias = True,
    #                             padding='same', data_format='channels_last', name='WD-0')(x_wd)

    x_wd = WaveletDeconvolution(WD_spec[0], kernel_length=WD_spec[1], strides=WD_spec[2], use_bias = False,
                                padding='same', data_format='channels_last', name='WD-0')(x_in)
    x_wd = BatchNormalization(axis=-1)(x_wd)
    x_wd = Dropout(droprate)(x_wd)
    # x_wd = Activation(depth_activation)(x_wd)
    # x_wd  = SpatialDropout2D(0.2)(x_wd)
    # xin_expand = Lambda(lambda y: y[:,:,None,:])(x_in)
    # x_wd = Concatenate(axis=2)([x_wd, xin_expand])
    #The C - F branch ==================================
    x = DepthwiseConv2D((1, WD_spec[0]), strides=(1, 1), padding="valid",
                        depth_multiplier=depth_multiplier,
                        data_format=None, dilation_rate=(1, 1), 
                        activation=None,  use_bias=False,
                        depthwise_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        depthwise_regularizer=None, bias_regularizer=None,
                        activity_regularizer=None, depthwise_constraint=max_norm(normrate_head),
                        bias_constraint=None, 
                        name = 'spatialconv')(x_wd)
    x = BatchNormalization(axis=-1)(x)
    x = SpatialDropout2D(0.3)(x)
    x = Activation(depth_activation, name='spatialconv-act')(x)
    
    x = Lambda(lambda y: y[:, :, 0, :])(x)   
    x = SeparableConv1D( num_filters_list[0], kernel_size=kernel_size_list[0], strides=1, padding="valid",
                         data_format=None, dilation_rate=1, depth_multiplier=depth_rate*depth_multiplier,
                         activation=sep_activation, use_bias=True, depthwise_initializer="glorot_uniform",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None,
                         name = 'sepconv1-1'
                         )(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Dropout(droprate)(x)
    
    
    # The F - C branch =======================================================
    xt = Permute((1,3,2))(x_wd)
    # xt = SpatialDropout2D(0.2)(xt)
    # xt = GaussianDropout(0.5)(xt)

    xt = DepthwiseConv2D((1, in_shape[-1]), strides=(1, 1), padding="valid",
                         depth_multiplier=depth_multiplier,
                         activation=None,  use_bias=False,
                         depthwise_initializer="glorot_uniform",
                         bias_initializer="zeros",
                         depthwise_constraint=max_norm(normrate_head),
                         name = 'freqconv')(xt)
    xt = BatchNormalization(axis=-1)(xt)
    xt = SpatialDropout2D(0.3)(xt)
    xt = Activation(depth_activation, name='freqconv-act')(xt)
    xt = Lambda(lambda y: y[:, :, 0, :])(xt)
    
    xt = SeparableConv1D( num_filters_list[0], kernel_size=kernel_size_list[0], strides=1, padding="valid",
                         data_format=None, dilation_rate=1, depth_multiplier=depth_rate*depth_multiplier,
                         activation=sep_activation, use_bias=True, depthwise_initializer="glorot_uniform",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None,
                         name = 'sepconv2-1'
                         )(xt)
    
    # xt = Dropout(droprate)(xt)
    # Merge two branches =====================================================
    
    x = Add(name='merge')([x, xt])   
    # x = Concatenate(name='merge')([x, xt])
    # x = Multiply(name='merge')([x, xt])
    # x = Weight()([x, xt])
    # x = C_Weight()([x, xt])
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    # x = Dropout(droprate)(x)
    x = GaussianDropout(0.5)(x)
    
    # x = SeparableConv1D(num_filters_list[1], kernel_size=kernel_size_list[1], 
    #                     # depthwise_constraint=max_norm(normrate_head/2),
    #                     # activity_regularizer=l1(1e-5),
    #                     activation=None,use_bias = True, name='sepconv-{}'.format(1))(x)
    # # x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    
    for i in range(len(strides_for_pool)):
        x = SeparableConv1D(num_filters_list[1+i], kernel_size=kernel_size_list[1+i], 
                            # depthwise_constraint=max_norm(normrate_head/2),
                            # activity_regularizer=l1(1e-5),
                            depthwise_initializer="glorot_normal",
                            pointwise_initializer="glorot_uniform",
                            activation=None,use_bias = True, name='sepconv-{}'.format(i+1))(x)
        # x = BatchNormalization(axis=-1)(x)
        x = Activation('elu')(x)
        x = SeparableConv1D(num_filters_list[1+i], kernel_size=kernel_size_list[1+i], strides = strides_for_pool[i],
                            # depthwise_constraint=max_norm(normrate_head/2),
                            # activity_regularizer=l1(1e-5),
                            depthwise_initializer="glorot_normal",
                            pointwise_initializer="glorot_uniform",
                            activation=None, use_bias = True, name='pooling-{}'.format(i+1))(x)
        x = BatchNormalization(axis=-1)(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('elu')(x)

    
    # x = SeparableConv1D(2*num_filters_list[-1], kernel_size=3, use_bias = True, name='sepconv-end')(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    
    x = Dropout(droprate)(x)
    
    x = Flatten(name = 'flatten')(x)
    # x = GlobalAveragePooling1D(name = 'flatten')(x)
    
    # x = Dense(32, name='feature', activation = 'elu')(x)
    
    x = Dense(num_classes, name = 'dense', kernel_constraint = max_norm(normrate_dense) )(x)
    
    softmax = Activation('softmax', name = 'softmax')(x)
    
    Mymodel = Model(x_in, softmax)
    Mymodel.compile(loss='categorical_crossentropy', 
                    # loss = damped_log_loss,
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    
    return Mymodel

#%%
def TFCNet(in_shape, num_classes, 
           depth_activation = 'tanh', sep_activation = 'linear',
           depth_multiplier = 8, depth_rate=2,
           num_filters_list = [16, 32, 64], kernel_size_list=[5,5,5],
           strides_for_pool=[5,5],
           optimizer=Adam, learning_rate=1e-3, droprate=0.5, normrate_head=1.0, normrate_dense = 0.5):
    '''
    in_shape = (time, channel, freqband)
    '''
    
    x_in = Input(shape = in_shape, name = 'input')
    # x = GaussianDropout(0.5)(x_in)
    x = Dropout(0.5)(x_in)

    
    #The C - F branch ==================================
    x = DepthwiseConv2D((1, in_shape[1]), strides=(1, 1), padding="valid",
                        depth_multiplier=depth_multiplier,
                        data_format=None, dilation_rate=(1, 1), 
                        activation=None,  use_bias=False,
                        depthwise_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        depthwise_regularizer=None, bias_regularizer=None,
                        activity_regularizer=None, depthwise_constraint=max_norm(normrate_head),
                        bias_constraint=None, 
                        name = 'spatialconv')(x)
    x = BatchNormalization(axis=-1)(x)
    x = SpatialDropout2D(0.3)(x)
    x = Activation(depth_activation, name='spatialconv-act')(x)
    
    # x = Permute((-2,-1))(x)
    # x = Reshape((-1, in_shape[-1]))(x)
    x = Lambda(lambda y: y[:, :, 0, :])(x)
    
    x = SeparableConv1D( num_filters_list[0], kernel_size=kernel_size_list[0], strides=1, padding="valid",
                         data_format=None, dilation_rate=1, depth_multiplier=depth_rate*depth_multiplier,
                         activation=sep_activation, use_bias=True, depthwise_initializer="glorot_uniform",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None,
                         name = 'sepconv1-1'
                         )(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Dropout(droprate)(x)
    
    
    # The F - C branch =======================================================
    xt = Permute((1,3,2))(x_in)
    xt = GaussianDropout(0.5)(xt)

    xt = DepthwiseConv2D((1, in_shape[-1]), strides=(1, 1), padding="valid",
                         depth_multiplier=depth_multiplier,
                         activation=None,  use_bias=False,
                         depthwise_initializer="glorot_uniform",
                         bias_initializer="zeros",
                         depthwise_constraint=max_norm(normrate_head),
                         name = 'freqconv')(xt)
    xt = BatchNormalization(axis=-1)(xt)
    xt = SpatialDropout2D(0.3)(xt)
    xt = Activation(depth_activation, name='freqconv-act')(xt)
    # x = Permute((-2,-1))(x)
    # xt = Reshape((-1, in_shape[-1]))(xt)
    xt = Lambda(lambda y: y[:, :, 0, :])(xt)
    
    xt = SeparableConv1D( num_filters_list[0], kernel_size=kernel_size_list[0], strides=1, padding="valid",
                         data_format=None, dilation_rate=1, depth_multiplier=depth_rate*depth_multiplier,
                         activation=sep_activation, use_bias=True, depthwise_initializer="glorot_uniform",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None,
                         name = 'sepconv2-1'
                         )(xt)
    
    # xt = Dropout(droprate)(xt)
    # Merge two branches =====================================================
    
    x = Add(name='merge')([x, xt])   
    # x = Concatenate(name='merge')([x, xt])
    # x = Multiply(name='merge')([x, xt])
    # x = Weight()([x, xt])
    # x = C_Weight()([x, xt])
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    # x = Dropout(droprate/4)(x)
    xt = GaussianDropout(0.5)(xt)
    
    for i in range(len(strides_for_pool)):
        x = SeparableConv1D(num_filters_list[1+i], kernel_size=kernel_size_list[1+i], 
                            # depthwise_constraint=max_norm(normrate_head/2),
                            # activity_regularizer=l1(1e-5),
                            activation=None,use_bias = True, name='sepconv-{}'.format(i+1))(x)
        # x = BatchNormalization(axis=-1)(x)
        x = Activation('elu')(x)
        x = SeparableConv1D(num_filters_list[1+i], kernel_size=kernel_size_list[1+i], strides = strides_for_pool[i],
                            # depthwise_constraint=max_norm(normrate_head/2),
                            # activity_regularizer=l1(1e-5),
                            activation=None, use_bias = True, name='pooling-{}'.format(i+1))(x)
        x = BatchNormalization(axis=-1)(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('elu')(x)
       
# =============================================================================
#     x = SeparableConv1D(num_filters_list[1], kernel_size=kernel_size_list[1], 
#                         activation=None,use_bias = True, name='sepconv-1')(x)
#     # x = BatchNormalization(axis=1)(x)
#     x = Activation('elu')(x)
#     x = SeparableConv1D(num_filters_list[1], kernel_size=kernel_size_list[1], strides = strides_for_pool[0],
#                         activation=None, use_bias = True, name='pooling-1')(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('elu')(x)
#     
#     x = SeparableConv1D(num_filters_list[2], kernel_size=kernel_size_list[2], 
#                         activation=None, use_bias = True, name='sepconv-2')(x)
#     # x = BatchNormalization(axis=1)(x)
#     x = Activation('elu')(x)
#     x = SeparableConv1D(num_filters_list[2], kernel_size=kernel_size_list[2], strides = strides_for_pool[1],
#                         activation=None, use_bias = True, name='pooling-2')(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('elu')(x)
# =============================================================================
    
    # x = SeparableConv1D(num_filters_list[2], kernel_size=1, use_bias = True, name='sepconv-3')(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Activation('elu')(x)
    
    x = Dropout(droprate)(x)
    
    x = Flatten(name = 'flatten')(x)
    # x = GlobalAveragePooling1D(name = 'flatten')(x)
    
    # x = Dense(32, name='feature', activation = 'elu')(x)
    
    x = Dense(num_classes, name = 'dense', kernel_constraint = max_norm(normrate_dense) )(x)
    
#    dense        = add([dense, dense1])
    softmax = Activation('softmax', name = 'softmax')(x)
    
    Mymodel = Model(x_in, softmax)
    Mymodel.compile(loss='categorical_crossentropy', 
                    # loss = damped_log_loss,
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel

#%%
def TFCNet_1(in_shape, num_classes,
             depth_activation = 'tanh',
             num_filters_list = [16, 32, 64], kernel_size_list=[5,5,5],
             strides_for_pool=[5,5],
             optimizer=Adam, learning_rate=1e-3, droprate=0.5, normrate_head=1.0, normrate_dense = 0.5):
    '''
    in_shape = (time, channel, freqband)
    '''
    
    x_in = Input(shape = in_shape, name = 'input')
    
    #The C - F branch ==================================
    x = DepthwiseConv2D((1, in_shape[1]), strides=(1, 1), padding="valid",
                        depth_multiplier=2,
                        data_format=None, dilation_rate=(1, 1), 
                        activation=depth_activation ,  use_bias=False,
                        depthwise_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        depthwise_regularizer=None, bias_regularizer=None,
                        activity_regularizer=None, depthwise_constraint=max_norm(normrate_head),
                        bias_constraint=None, 
                        name = 'spatialconv')(x_in)
    
    x = Permute((1,3,2))(x)
    x = DepthwiseConv2D((1, in_shape[-1]), strides=(1, 1), padding="valid",
                        depth_multiplier=2,
                        activation=depth_activation ,  use_bias=False,
                        depthwise_constraint=max_norm(normrate_head),
                        name = 'freqconv-2')(x)
 
    
    x = Lambda(lambda y: y[:, :, 0, :])(x)
    
    x = SeparableConv1D( num_filters_list[0], kernel_size=kernel_size_list[0], strides=1, padding="valid",
                         data_format=None, dilation_rate=1, depth_multiplier=2,
                         activation='elu', use_bias=True, depthwise_initializer="glorot_uniform",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None,
                         name = 'sepconv1-1'
                         )(x)
    # x = BatchNormalization(axis=1)(x)
    
    
    
    # The F - C branch =======================================
    xt = Permute((1,3,2))(x_in)

    xt = DepthwiseConv2D((1, in_shape[-1]), strides=(1, 1), padding="valid",
                         depth_multiplier=2,
                         activation=depth_activation ,  use_bias=False,
                         depthwise_initializer="glorot_uniform",
                         bias_initializer="zeros",
                         depthwise_constraint=max_norm(normrate_head),
                         name = 'freqconv')(xt)

    xt = Permute((1,3,2))(xt)
    xt = DepthwiseConv2D((1, in_shape[1]), strides=(1, 1), padding="valid",
                         depth_multiplier=2,
                         activation=depth_activation ,  use_bias=False,
                         depthwise_constraint=max_norm(normrate_head),
                         name = 'spatialconv-2')(xt) 
    
    
    xt = Lambda(lambda y: y[:, :, 0, :])(xt)
    
    xt = SeparableConv1D( num_filters_list[0], kernel_size=kernel_size_list[0], strides=1, padding="valid",
                         data_format=None, dilation_rate=1, depth_multiplier=2,
                         activation='elu', use_bias=True, depthwise_initializer="glorot_uniform",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None,
                         name = 'sepconv2-1'
                         )(xt)
    
    # Merge two branches ==============================================
    
    x = Add()([x, xt])
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    # x = Dropout(droprate)(x)
    
    # x = Conv1D(num_filters_list[1], kernel_size=kernel_size_list[1], use_bias = True, activation='elu', name='sepconv-0')(x)
    x = SeparableConv1D(num_filters_list[1], kernel_size=kernel_size_list[1], use_bias = True, name='sepconv-1')(x)
    x = Activation('elu')(x)
    x = SeparableConv1D(num_filters_list[1], kernel_size=kernel_size_list[1], strides = strides_for_pool[0], use_bias = True, name='pooling-1')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    
    x = SeparableConv1D(num_filters_list[2], kernel_size=kernel_size_list[2], use_bias = True, name='sepconv-2')(x)
    x = Activation('elu')(x)
    x = SeparableConv1D(num_filters_list[2], kernel_size=kernel_size_list[2], strides = strides_for_pool[1], 
                        use_bias = True, name='pooling-2')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    
    # x = SeparableConv1D(num_filters_list[2], kernel_size=kernel_size_list[2], use_bias = True, name='sepconv-3')(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Activation('elu')(x)
    
    x = Dropout(droprate)(x)
    
    x = Flatten(name = 'flatten')(x)
#    flatten      = GlobalAveragePooling1D(axis = 1)(block2)
    
#    dense        = Dense(32, kernel_constraint = max_norm(norm_rate), name='feature', activation = 'elu')(flatten)
    
    x = Dense(num_classes, name = 'dense', kernel_constraint = max_norm(normrate_dense) )(x)
    
#    dense        = add([dense, dense1])
    softmax = Activation('softmax', name = 'softmax')(x)
    
    Mymodel = Model(x_in, softmax)
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel

from tensorflow.keras.layers import Conv2D, AveragePooling2D, SeparableConv2D, MaxPooling2D

'''
From https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py 
for benchmark
'''

def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
             optimizer = Adam, learning_rate = 1e-3):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 
    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Samples, Chans, 1))

    ##################################################################
    block1       = Conv2D(F1, (kernLength, 1), padding = 'same',
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = -1)(block1)  # normalization on channels or time
    block1       = DepthwiseConv2D((1, Chans), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = -1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((2, 1))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (5, 1),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = -1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((2, 1))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    Mymodel      = Model(inputs=input1, outputs=softmax)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel

def WD_EEGNet(nb_classes, Chans = 64, Samples = 128, WD_spec = [8, 5, 1],
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
             optimizer = Adam, learning_rate = 1e-3):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Samples, Chans))
    ##################################################################
    x_wd     = WaveletDeconvolution(WD_spec[0], kernel_length=WD_spec[1], strides=WD_spec[2], use_bias = False,
                                    padding='same', data_format='channels_last', name='WD-0')( input1)
    x_wd     = Permute((1,3,2))(x_wd)
    ##################################################################
    block1       = Conv2D(F1, (kernLength, 1), padding = 'same',
                                   use_bias = False)(x_wd)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D((1, Chans), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((2, 1))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (5, 1),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((2, 1))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    Mymodel      = Model(inputs=input1, outputs=softmax)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel


def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5,
                optimizer = Adam, learning_rate = 1e-3):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    input_main   = Input((Samples, Chans, 1))
    block1       = Conv2D(8, (5, 1), 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(8, (1, Chans),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(16, (5, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(32, (5, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(64, (5, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    Mymodel      = Model(inputs=input_main, outputs=softmax)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel
#%%
'''
Generative Models
'''

# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Conv1DTranspose

# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z"""

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# def vae_static(in_shape, latent_dim, num_filters_list, kernel_size_list, strides_for_pool):
#     '''
#     in_shape: (time, channels)
#     '''
    
#     # encoder ================================================================
#     encoder_inputs = Input(shape=in_shape)
#     x = Conv1D(num_filters_list[0], kernel_size=kernel_size_list[0], use_bias = True, 
#                padding='same', activation='elu', name='encoder-head')(encoder_inputs)
    
#     for i in range(0, len(num_filters_list)):
#         x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], 
#                             padding='same', use_bias = True, activation = 'elu',
#                             name='encoder-{}'.format(i))(x)
#         x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i],
#                             activation = 'elu',
#                             strides = strides_for_pool[i], use_bias = True, 
#                             name='pooling-{}'.format(i))(x) 
#         x = BatchNormalization()(x)

    
#     x = Flatten()(x)
#     x = Dense(32, activation="elu")(x)
#     z_mean = Dense(latent_dim, name="z_mean")(x)
#     z_log_var = Dense(latent_dim, name="z_log_var")(x)
#     z = Sampling()([z_mean, z_log_var]) # sampling layer
#     encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
#     # decoder =================================================================
#     latent_inputs = Input(shape=(latent_dim,))
#     stride_rate = np.prod(strides_for_pool)
#     x = layers.Dense(in_shape[0]//stride_rate * num_filters_list[-1], activation="elu")(latent_inputs)
#     x = layers.Reshape((in_shape[0]//stride_rate , num_filters_list[-1]))(x)
    
#     for i in range(len(num_filters_list)-1, -1, -1):
#         x = Conv1D(num_filters_list[i], kernel_size_list[i], activation="elu", padding="same")(x)
#         x = Conv1DTranspose(num_filters_list[i], kernel_size_list[i], activation="elu", 
#                             strides=strides_for_pool[i], padding="same")(x)
#         x = BatchNormalization()(x)
        
#     decoder_outputs = Conv1DTranspose(in_shape[-1], num_filters_list[-1], activation="linear", padding="same")(x) # different activation, use point-wise conv ?
#     decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    
#     return encoder, decoder

# class VAE(Model):
#     def __init__(self, encoder, decoder, alpha, **kwargs):
#         super(VAE, self).__init__(**kwargs)
#         self.encoder = encoder
#         self.decoder = decoder
#         self.alpha=alpha

#     def train_step(self, data):
#         if isinstance(data, tuple):
#             data = data[0]
#         with tf.GradientTape() as tape:
#             z_mean, z_log_var, z = self.encoder(data)
#             reconstruction = self.decoder(z)
#             reconstruction_loss = tf.reduce_mean(
#                 # losses.binary_crossentropy(data, reconstruction)  # reconstruction loss
#                 losses.mean_absolute_error(data, reconstruction)
#             )
#             reconstruction_loss *= (data.shape[1]*data.shape[2])
#             kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)  # how to use kernel trick?
#             kl_loss = tf.reduce_mean(kl_loss)
#             kl_loss *= -0.5
#             total_loss = (reconstruction_loss + self.alpha*kl_loss)
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         return {
#             "loss": total_loss,
#             "reconstruction_loss": reconstruction_loss,
#             "kl_loss": kl_loss,
#         }



# def vae_lstm():
#     pass
    

# #%%
# if __name__ == '__main__':
#     # model = TFCNet((400, 10, 5), 4)
#     # model.summary()
#     # model_1 = TFCNet_1((400, 10, 5), 4)
#     # model_1.summary()
    
#     # encoder, decoder = vae_static(in_shape=(400, 10), 
#     #                               latent_dim=2, 
#     #                               num_filters_list=[16,16,16], 
#     #                               kernel_size_list=[3,3,3], 
#     #                               strides_for_pool=[4,2,2])
    
#     # encoder.summary()
#     # decoder.summary()
    
#     import pywt
#     low_pass, high_pass  = pywt.Wavelet("sym20").filter_bank[:2]
#     low_pass = np.array(low_pass)
#     high_pass = np.array(high_pass)
    
#     in_shape = (400, 10)        
#     WW = wavelet_net(in_shape,
#                      low_pass, high_pass,
#                      num_filters=1, strides = 2,
#                      target_size=100, min_pool_size =3, 
#                      n_levels = 3,
#                      wavelets_trainable = True)
#     print(WW.output)
    
    
    