import numpy as np
import keras
import json
from tqdm import tqdm
import cv2
import random
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as keras_image
import pickle

def augment_patch(patch, augmentation):

    if augmentation=='H-Flip':
        augmented_patch = np.fliplr(patch)
    elif augmentation=='V-Flip':
        augmented_patch = np.flipud(patch)
    elif augmentation=='180':
        augmented_patch = np.rot90(patch, 2)
    elif augmentation=='90':
        augmented_patch = np.rot90(patch, 1)
    elif augmentation=='270':
        augmented_patch = np.rot90(patch, 3)
    else:
        augmented_patch = patch

    return augmented_patch

def length(list_IDs, batch_size):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(list_IDs) / batch_size))

def load_image_paths(image_categories):

    with open('../train_pdfs.pkl', 'rb') as f:
        pdfs = pickle.load(f)

    with open('../from_scratch_classification.json', 'r') as f:
        cls_type = json.load(f)

    list_IDs = []
    for doi in tqdm(pdfs):
        for panel,cls in cls_type[doi].items():
            if cls in image_categories:
                path = '/nas/medifor/esabir/scientific_integrity/from_scratch/from_scratch_panels/'+doi+'/'+panel
                list_IDs += [path]
                    
    return list_IDs

def on_epoch_end(list_IDs):
    'Updates indexes after each epoch'
    indexes = np.arange(len(list_IDs))
    np.random.shuffle(indexes)
        
    return indexes

def get_batch(indexes, index, list_IDs, batch_size, resize_dim, dim, n_channels, augmentation_list):

    indexes = indexes[index*batch_size:(index+1)*batch_size]

    # Find list of IDs
    list_IDs_temp = [list_IDs[k] for k in indexes]
    
    # Generate data
    X, y, y1, y2 = data_generation(list_IDs_temp, batch_size, resize_dim, dim, n_channels, augmentation_list)

    return X, y, y1, y2

def create_spliced_manipulation(img, resize_dim, augmentation_list):

    img = cv2.resize(img, resize_dim)

    h, w, ch = img.shape

    new_h, new_w = int(np.ceil(h/16.)*16), int(np.ceil(w/16.)*16)

    new_img = np.zeros((new_h, new_w, ch))
    new_img[:h,:w,:] = img

    mask_img = np.zeros_like(new_img)

    duplicate = True

    if duplicate:

        dup1_r1 = random.randint(0,np.floor(0.75*new_h))
        dup1_c1 = random.randint(0,np.floor(0.75*new_w))
        dup1_r2 = random.randint(dup1_r1+10, dup1_r1+np.floor(0.25*new_h))
        dup1_c2 = random.randint(dup1_c1+10, dup1_c1+np.floor(0.25*new_w))

        assert np.floor(0.75*new_h)-dup1_r1>=0, 'Negative row for second patch!'
        assert np.floor(0.75*new_w)-dup1_c1>=0, 'Negative col for second patch!'

        augmentation = random.choice(augmentation_list)

        dup2_r1 = random.randint(0, np.floor(0.75*new_h))
        dup2_c1 = random.randint(0, np.floor(0.75*new_w))

        if augmentation in ['0', '180', 'H-Flip', 'V-Flip']:
            dup2_r2 = dup2_r1 + (dup1_r2-dup1_r1)
            dup2_c2 = dup2_c1 + (dup1_c2-dup1_c1)
        else:
            dup2_r2 = dup2_r1 + (dup1_c2-dup1_c1)
            dup2_c2 = dup2_c1 + (dup1_r2-dup1_r1)

        assert dup2_r2<=new_h, 'Second patch row out of bounds!'
        assert dup2_c2<=new_w, 'Second patch col out of bounds!'

        #if random.choice([True, False]):
        #    patch = new_img[dup2_r1:dup2_r2,dup2_c1:dup2_c2,:]
        #    augmented_patch = augment_patch(patch, augmentation)
        #    new_img[dup1_r1:dup1_r2,dup1_c1:dup1_c2,:] = augmented_patch
        #else:
        patch = new_img[dup1_r1:dup1_r2,dup1_c1:dup1_c2,:]
        augmented_patch = augment_patch(patch, augmentation)
        new_img[dup2_r1:dup2_r2,dup2_c1:dup2_c2,:] = augmented_patch

        dup_coord1 = (dup1_r1,dup1_r2,dup1_c1,dup1_c2)
        dup_coord2 = (dup2_r1,dup2_r2,dup2_c1,dup2_c2)

        mask_img[dup1_r1:dup1_r2,dup1_c1:dup1_c2,1] = 1
        mask_img[dup2_r1:dup2_r2,dup2_c1:dup2_c2,0] = 1
        mask_img[:,:,2] = 1
        tmp = mask_img[:,:,1] + mask_img[:,:,0]
        tmp[tmp>0] = 1
        mask_img[:,:,2] = mask_img[:,:,2] - tmp

        simi_mask = np.concatenate((mask_img[:,:,:1]+mask_img[:,:,1:2], mask_img[:,:,2:3]), axis=-1)
        mani_mask = np.concatenate((mask_img[:,:,:1], mask_img[:,:,1:2]+mask_img[:,:,2:3]), axis=-1)

    return new_img, mask_img, simi_mask, mani_mask

def unmanipulated(img, resize_dim):

    img = cv2.resize(img, resize_dim)
    
    gt_mask = np.zeros_like(img)
    gt_mask[:,:,2] = 1
    
    return img, gt_mask, gt_mask[:,:,:2], gt_mask[:,:,:2]

def create_manipulation(img, resize_dim, augmentation_list):
        
    choices = ['Pristine', 'Splice']
    choice = random.choice(choices)

    if choice=='Pristine':
        img, gt_mask, gt_mask1, gt_mask2 = unmanipulated(img, resize_dim)
    elif choice=='Splice':
        img, gt_mask, gt_mask1, gt_mask2 = create_spliced_manipulation(img, resize_dim, augmentation_list)
    else:
        print('Invalid choice!')
        raise SystemExit

    img = img[:,:,::-1]

    return img, gt_mask, gt_mask1, gt_mask2

def data_generation(list_IDs_temp, batch_size, resize_dim, dim, n_channels, augmentation_list):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((batch_size, dim[0], dim[1], n_channels))
    y = np.empty((batch_size, dim[0], dim[1], 3))
    y1 = np.empty((batch_size, dim[0], dim[1], 2))
    y2 = np.empty((batch_size, dim[0], dim[1], 2))
        
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        img = cv2.imread(ID)
        X[i], y[i], y1[i], y2[i] = create_manipulation(img, resize_dim, augmentation_list)

    return X, y, y1, y2

def DataGenerator(image_categories, augmentation_list):
    'Generates data for Keras'

    dim = (256,256)
    resize_dim = (256, 256)
    batch_size = 32
    list_IDs = load_image_paths(image_categories)
    n_channels = 3
    indexes = on_epoch_end(list_IDs)

    while True:

        for index in range(length(list_IDs, batch_size)):
            X, y, y1, y2 = get_batch(indexes, index, list_IDs, batch_size, resize_dim, dim, n_channels, augmentation_list)
            yield (X,y)

        indexes = on_epoch_end(list_IDs)
