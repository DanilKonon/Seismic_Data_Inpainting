#!/usr/bin/env python
# coding: utf-8

import sys
import io
import json
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from pathlib import Path
import cv2
import skimage
from tensorflow.keras.applications import ResNet50V2, ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, BatchNormalization, Conv2D, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from adamp_tf import AdamP
from sgdp_tf import SGDP
from collections import Callable
import time
from tensorflow.keras import backend as K


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

from skimage.transform import resize
import albumentations as A

def augment_img_mask(x, y):
    transform = A.Compose(
        [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(p=0.5, alpha=240, sigma=240 * 0.05, alpha_affine=240 * 0.03)
        ]
    )

    transform_image = transform(image=x, mask=y)
    return transform_image['image'], transform_image['mask']


class DataGeneratorDivide(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path_to_dataset, batch_size=32, 
                 shuffle=True, use_augmentations=False,
                 mode='train', val_percent=0.3):
        """
        mode: train or val
        """
        self.batch_size = batch_size
        self.path_to_dataset = path_to_dataset
        self.val_percent = val_percent
        self.mode = mode
        self.initialize()
        
        self.shuffle = shuffle
        self.on_epoch_end()
        self.use_aug = use_augmentations

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y = self.__data_generation(indexes)

        return X, Y
     
    def initialize(self):
        slice_nums = list(set(
            int(file.name.split('_')[-1].split('.')[0]) for file in (self.path_to_dataset / 'gt').iterdir()
        ))
        slice_nums = sorted(list(slice_nums))
        num_of_slices = len(slice_nums)
        val_num = int(num_of_slices * self.val_percent)
        if self.mode == 'train':
            curr_slices_to_use = slice_nums[val_num:]
        else:
            curr_slices_to_use = slice_nums[:val_num]
            
        self.curr_slices_to_use = curr_slices_to_use
        
        self.X, self.Y = [], []
        for file in (self.path_to_dataset / 'images').iterdir():
            slice_num = int(file.name.split('_')[-1].split('.')[0])
            if slice_num in self.curr_slices_to_use:
                self.X.append(file)
                self.Y.append(self.path_to_dataset / 'gt' / file.name)
                   
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # Resize or padd?
        # imread (M, N, 3). can take only first dim to make them grey
        # but resnet preprocessing wants rgb images!!!
        X = [np.load(self.X[ind]) for ind in indexes]
        Y = [np.load(self.Y[ind]) for ind in indexes]
        # batch_shapes = [el.shape for el in X]
        max_w, max_h = 256, 256
        # print(max_w, max_h)
        # # Generate data
        for i, img in enumerate(X):
            w, h = X[i].shape
            X[i] = resize(X[i], (256, 256), preserve_range=True) 
            Y[i] = resize(Y[i], (256, 256), preserve_range=True) 
            if self.use_aug:
                X[i], Y[i] = augment_img_mask(X[i], Y[i])
                # y[i] = y[i][:, :, np.newaxis]
            # X[i] = (np.pad(X[i], pad_width=((0, max_w - w), (0, max_h - h), (0, 0)))) 
            
#             X[i] = tf.keras.applications.resnet.preprocess_input(X[i]) 
            
            # np.pad(y[i], pad_width=((0, max_w - w), (0, max_h - h), (0, 0)))
            # X[i], y[i] = np.pad()
        X, Y = np.array(X)[:, :, :, np.newaxis], np.array(Y)[:, :, :, np.newaxis]
#         X_padded = np.zeros([X.shape[0], 512, 512, 1])
#         X_padded[:, :X.shape[1], :X.shape[2], :] = X
#         Y_padded = np.zeros([Y.shape[0], 512, 512, 1])
#         Y_padded[:, :Y.shape[1], :Y.shape[2], :] = Y
        return X, Y


def get_model(
        weight_decay=0.0001,
        start_neuron_number=16
    ):
    keras.backend.clear_session()

    wd_reg = l2(weight_decay)

    inputs = Input((256, 256, 1))
    x = inputs
    # s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(start_neuron_number * 1, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (x)
    # c1 = Dropout(0.1) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(start_neuron_number * 1, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(start_neuron_number * 2, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (p1)
    # c2 = Dropout(0.1) (c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(start_neuron_number * 2, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(start_neuron_number * 4, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (p2)
    # c3 = Dropout(0.2) (c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(start_neuron_number * 4, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(start_neuron_number * 8, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (p3)
    # c4 = Dropout(0.2) (c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(start_neuron_number * 8, (3, 3), activation='relu', kernel_initializer='he_normal',kernel_regularizer=wd_reg,  padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    # p4 = p3

    c5 = Conv2D(start_neuron_number * 8, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (p4)
    # c5 = Dropout(0.3) (c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(start_neuron_number * 8, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (c5)

    u6 = Conv2DTranspose(start_neuron_number * 8, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=wd_reg) (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(start_neuron_number * 8, (3, 3), activation='relu', kernel_initializer='he_normal',kernel_regularizer=wd_reg,  padding='same') (u6)
    # c6 = Dropout(0.2) (c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(start_neuron_number * 8, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (c6)

    u7 = Conv2DTranspose(start_neuron_number * 4, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=wd_reg) (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(0.2)(u7)
    c7 = Conv2D(start_neuron_number * 4, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (u7)
    # c7 = BatchNormalization()(c7)
    c7 = Conv2D(start_neuron_number * 4, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (c7)

    u8 = Conv2DTranspose(start_neuron_number * 2, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=wd_reg) (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.2)(u8)
    c8 = Conv2D(start_neuron_number * 2, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (u8)
    # c8 = BatchNormalization()(c8)
    c8 = Conv2D(start_neuron_number * 2, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (c8)

    u9 = Conv2DTranspose(start_neuron_number, (4, 4), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(0.2)(u9)
    c9 = Conv2D(start_neuron_number, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(start_neuron_number, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=wd_reg, padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='linear', dtype='float32') (c9)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def l1(y_true, y_pred):
    #print(y_true)
    #print(y_pred)
    """Calculate the L1 loss used in all loss calculations"""
    if K.ndim(y_true) == 4:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
    elif K.ndim(y_true) == 3:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2])
    else:
        raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")


def compute_perceptual(vgg_out, vgg_gt):
    """Perceptual loss based on VGG16, see. eq. 3 in paper"""       
    loss = 0
    for o, g in zip(vgg_out, vgg_gt):
        loss += l1(o, g)
    return loss


def gram_matrix(x, norm_by_channels=False):
    """Calculate gram matrix used in style loss"""

    # Assertions on input
#     print(K.ndim(x), x.shape)
    assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
    assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
    #import pdb
    #pdb.set_trace()
    # Permute channels and get resulting shape
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    shape = K.shape(x)
    B, C, H, W = shape[0], shape[1], shape[2], shape[3]

    # Reshape x and do batch dot product
    features = K.reshape(x, K.stack([B, C, H*W]))
    gram = K.batch_dot(features, features, axes=2)

    # Normalize with channels, height and width
    gram = gram /  K.cast(C * H * W, x.dtype)

    return gram


def compute_style(vgg_out, vgg_gt):
    """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
    loss = 0
    for o, g in zip(vgg_out, vgg_gt):
        loss += l1(gram_matrix(o), gram_matrix(g))
    return loss


def get_extracted_values(feature_extractor, y_true, y_pred):
    vgg_out = feature_extractor(y_true)
    vgg_gt = feature_extractor(y_pred)

    if not isinstance(vgg_out, list):
        vgg_out = [vgg_out]
        vgg_gt = [vgg_gt]

    # TODO: make output of autoencoder float32 / это же слои!!! я не  смогу так сделать
    vgg_out_ = []
    vgg_gt_ = []
    for el1, el2 in zip(vgg_out, vgg_gt):
        vgg_out_.append(K.cast(el1, 'float32'))
        vgg_gt_.append(K.cast(el2, 'float32'))

    vgg_gt = vgg_gt_
    vgg_out = vgg_out_
    return vgg_gt, vgg_out


def compute_loss_tv(P):
    # Calculate total variation loss
    a = l1(P[:,1:,:,:], P[:,:-1,:,:])
    b = l1(P[:,:,1:,:], P[:,:,:-1,:])        
    return a+b


def loss_total(
        feature_extractor_content,
        feature_extractor_style
    ):
    """
    Creates a loss function which sums all the loss components 
    and multiplies by their weights. See paper eq. 7.
    """
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        
        # Here I assume that rectangular shape is always the same
        mask = np.zeros(y_true.shape)
        xmin, xmax, ymin, ymax = (55, 200, 86, 169)
        mask[:, xmin-20:xmax+20, ymin-20:ymax+20] = 1
        mask = K.cast(mask, 'float32')
        
        vgg_gt_c, vgg_out_c = get_extracted_values(feature_extractor_content, y_true, y_pred)
        vgg_gt_s, vgg_out_s = get_extracted_values(feature_extractor_style, y_true, y_pred)
        
        loss_mae_hole = l1(mask * y_true, mask * y_pred)
        loss_mae_valid = l1((1 - mask) * y_true, (1 - mask) * y_pred)
        
        loss_perceptual = compute_perceptual(vgg_out_c, vgg_gt_c)
        loss_style = compute_style(vgg_out_s, vgg_gt_s)
        loss_tv_val = compute_loss_tv(P=mask * y_pred)

        # Return loss function
        return loss_mae_valid, loss_mae_hole, loss_perceptual, loss_style, loss_tv_val

    return loss


def make_linear_lr(min_lr, max_lr, number_of_steps):
    def gen_lr(step):
        return (max_lr - min_lr) / number_of_steps * step + min_lr
    return gen_lr


def make_cosine_anneal_lr(learning_rate, alpha, decay_steps):
    def gen_lr(global_step):
        global_step = tf.minimum(global_step, decay_steps)
        global_step = tf.cast(global_step, tf.float32)
        cosine_decay = 0.5 * (1 + tf.math.cos(3.1415926 * global_step / decay_steps)) # changed np.pi to 3.14
        decayed = (1 - alpha) * cosine_decay + alpha
        decayed_learning_rate = learning_rate * decayed
        return decayed_learning_rate
    return gen_lr


def make_cosine_annealing_with_warmup(min_lr, max_lr, number_of_steps, alpha, decay_steps):
    gen_lr_1 = make_linear_lr(min_lr, max_lr, number_of_steps)
    gen_lr_2 = make_cosine_anneal_lr(max_lr, alpha, decay_steps)
    def gen_lr(global_step):
        a = global_step < number_of_steps
        a = tf.cast(a, tf.float32)
        b = 1. - a
        return a * gen_lr_1(global_step) + b * gen_lr_2(global_step - number_of_steps)
        
    return gen_lr

class CosineAnnealingWithWarmUP(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, min_lr, max_lr, number_of_steps, alpha, decay_steps):
        super(CosineAnnealingWithWarmUP, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.number_of_steps = number_of_steps
        self.alpha = alpha
        self.decay_steps = decay_steps
        self.gen_lr_ca =  make_cosine_annealing_with_warmup(min_lr, max_lr, number_of_steps, alpha, decay_steps)
  
    def __call__(self, step):
        return self.gen_lr_ca(step)
    
    def get_config(self):
        config = {
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'number_of_steps': self.number_of_steps,
            'alpha': self.alpha,
            'decay_steps': self.decay_steps
            }
        return config


def choose_optimizer(
        optimizer_name='Adam',
        learning_rate_fn=0.001
    ):
    # (learning_rate=learning_rate_fn)
    if optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam
    elif optimizer_name == 'SGD':
        optimizer = tf.keras.optimizers.SGD
    elif optimizer_name == 'AdamP':
        optimizer = AdamP
    else:
        print('Choosing SGDP')
        optimizer = SGDP

    optimizer_with_lr = optimizer(learning_rate_fn)
    return optimizer_with_lr

def choose_learning_rate_func(
        type_lr_func='constant', max_lr = 0.001, 
        warmup_steps = 900, max_number_of_steps = 60_000,
        epochs=60
    ):
    if type_lr_func == 'constant':
        return max_lr
    else:
        return CosineAnnealingWithWarmUP(.0000001, max_lr, warmup_steps, 0, max_number_of_steps)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def main(params):
    weight_decay = params['weight_decay']
    start_neuron_number = params['start_neuron_number']
    optimizer_name = params['optimizer_name']
    type_lr_func = params['type_lr_func']
    max_lr = params['max_lr']
    warmup_steps = params['warmup_steps']
    max_number_of_steps = params['max_number_of_steps']
    epochs = params['epochs']
    save_model_tensorboard = params['save_model_tensorboard']
    style_layer_names = params['style_layer_names']
    content_layer_name = params['content_layer_name']
    mae_valid_weight = params['mae_valid_weight']
    mae_hole_weight = params['mae_hole_weight']
    perceptual_weight = params['perceptual_weight']
    style_weight = params['style_weight']
    tv_weight = params['tv_weight']

    model = get_model(weight_decay, start_neuron_number)
    path_to_dataset = Path('./dataset')
    autoencoder = tf.keras.models.load_model('./best_weights_24.h5', compile=False)
    autoencoder.trainable = False

    feature_extractor_style = keras.Model(
        inputs=autoencoder.input,
        outputs=[autoencoder.get_layer(l).output for l in style_layer_names]
    )

    feature_extractor_content = keras.Model(
        inputs=autoencoder.input,
        outputs=[autoencoder.get_layer(content_layer_name).output]
    )

    optimizer = choose_optimizer(
        optimizer_name, 
        choose_learning_rate_func(type_lr_func, max_lr, warmup_steps, max_number_of_steps, epochs)
    )

    dg_train = DataGeneratorDivide(
        path_to_dataset, mode='train', 
        val_percent=0.2, use_augmentations=True, 
        batch_size=6
    )

    dg_val = DataGeneratorDivide(path_to_dataset, mode='val', val_percent=0.2, batch_size=6)

    writer = tf.summary.create_file_writer(save_model_tensorboard)

    global_step = 0
    for ind in range(epochs):
        model.save(f'./{save_model_tensorboard}.h5')
        print(f'{ind} epoch')
        dg_train.on_epoch_end()
        
        for ind, (x, y) in enumerate(dg_val):
            if ind == 1:
                break
            prediction = model.predict(x)
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            for pred, x_, y_ in zip(prediction, x, y):
                axes[0].imshow(pred, cmap='gray')
                axes[1].imshow(x_, cmap='gray')
                axes[2].imshow(y_, cmap='gray')
    #         plt.show()
            with writer.as_default():
                tf.summary.image("Val data", plot_to_image(fig), step=global_step)
        
        start = time.time()
        for step_num, (inputs, targets) in enumerate(dg_train):
            global_step += 1
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                func = loss_total(feature_extractor_content, feature_extractor_style)
                loss_value_list = func(targets, predictions)
                loss_value =\
                    mae_valid_weight * loss_value_list[0] +\
                    mae_hole_weight * loss_value_list[1] +\
                    perceptual_weight * loss_value_list[2] +\
                    style_weight * loss_value_list[3] +\
                    tv_weight * loss_value_list[4]
                
            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            if step_num % 10 == 0:
                with writer.as_default():
                    tf.summary.scalar("loss_train", loss_value.numpy().mean(), step=global_step)
                    tf.summary.scalar("loss_train_mae_valid", loss_value_list[0].numpy().mean(), step=global_step)
                    tf.summary.scalar("loss_train_mae_hole", loss_value_list[1].numpy().mean(), step=global_step)
                    tf.summary.scalar("loss_train_percept" , loss_value_list[2].numpy().mean(), step=global_step)
                    tf.summary.scalar("loss_train_style", loss_value_list[3].numpy().mean(), step=global_step)
                    tf.summary.scalar("loss_train_tv", loss_value_list[4].numpy().mean(), step=global_step)
                    if isinstance(optimizer.lr, Callable):
                        cur_lr = optimizer.lr(global_step).numpy()
                    else:
                        cur_lr = optimizer.lr.numpy()
                    tf.summary.scalar("learning_rate", cur_lr, step=global_step)
                    writer.flush()
        end = time.time()
        print(f'Training took {end - start}')
        
        start = time.time()
        val_loss_value = 0
        corr_coef_value = 0
        batch_num = 0
        for step_num, (inputs, targets) in enumerate(dg_val):
            predictions = model(inputs)
            
            corr_coefs = []
            for pred, x_, y_ in zip(predictions, inputs, targets):
                xmin, xmax = min(np.where(x_ < 0.001)[0]), max(np.where(x_ < 0.001)[0])
                ymin, ymax = min(np.where(x_ < 0.001)[1]), max(np.where(x_ < 0.001)[1])
                y_ = y_[xmin-10:xmax+10, ymin-10:ymax+10]
                pred = pred[xmin-10:xmax+10, ymin-10:ymax+10]
                corr_coef = np.corrcoef(y_.ravel(), pred.numpy().ravel())[0, 1]
                corr_coefs.append(corr_coef)    
            corr_coef_value += np.mean(corr_coefs)
            
            func = loss_total(feature_extractor_content, feature_extractor_style)
            loss_value_list = func(targets, predictions)
            loss_value =\
                    mae_valid_weight * loss_value_list[0] +\
                    mae_hole_weight * loss_value_list[1] +\
                    perceptual_weight * loss_value_list[2] +\
                    style_weight * loss_value_list[3] +\
                    tv_weight * loss_value_list[4]
            val_loss_value += loss_value.numpy().mean() 
            batch_num += 1
            
        with writer.as_default():
            tf.summary.scalar("loss_val", val_loss_value / batch_num, step=global_step)
            tf.summary.scalar("corr_coeff_val", corr_coef_value / batch_num, step=global_step)
            writer.flush()
        end = time.time()   
        print(f'Val took {end - start}')
    


if __name__ == '__main__':
    path_to_json = sys.argv[1]
    with open(path_to_json, 'r') as f:
        params = json.load(f)
    main(params)
