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

def load_image_paths():

    with open('../train_pdfs.pkl', 'rb') as f:
        pdfs = pickle.load(f)

    with open('../from_scratch_classification.json', 'r') as f:
        cls_type = json.load(f)

    list_IDs = []
    for doi in tqdm(pdfs):
        for panel,cls in cls_type[doi].items():
            if cls in ['Blot/Gel']:
                path = '/nas/medifor/esabir/scientific_integrity/from_scratch/from_scratch_panels/'+doi+'/'+panel
                list_IDs += [path]
                    
    return list_IDs

def on_epoch_end(list_IDs):
    'Updates indexes after each epoch'
    indexes = np.arange(len(list_IDs))
    np.random.shuffle(indexes)
        
    return indexes

def get_batch(indexes, index, list_IDs, batch_size, resize_dim, dim, n_channels):

    indexes = indexes[index*batch_size:(index+1)*batch_size]

    # Find list of IDs
    list_IDs_temp = [list_IDs[k] for k in indexes]
    
    # Generate data
    X, y = data_generation(list_IDs_temp, batch_size, resize_dim, dim, n_channels)

    assert len(X.shape)==4, 'Input X mismatch!'
    assert len(y.shape)==4, 'Output y mismatch!'

    return X, y

def create_patch_manipulation(img, resize_dim):

    img = cv2.resize(img, resize_dim)

    h, w, ch = img.shape

    new_h, new_w = int(np.ceil(h/16.)*16), int(np.ceil(w/16.)*16)

    new_img = np.zeros((new_h, new_w, ch))
    new_img[:h,:w,:] = img

    mask_img = np.zeros_like(new_img[:,:,:1])

    duplicate = True

    if duplicate:

        dup1_r1 = random.randint(0,np.floor(0.75*new_h))
        dup1_c1 = random.randint(0,np.floor(0.75*new_w))
        dup1_r2 = random.randint(dup1_r1+5, dup1_r1+np.floor(0.25*new_h))
        dup1_c2 = random.randint(dup1_c1+5, dup1_c1+np.floor(0.25*new_w))

        assert np.floor(0.75*new_h)-dup1_r1>=0, 'Negative row for second patch!'
        assert np.floor(0.75*new_w)-dup1_c1>=0, 'Negative col for second patch!'

        augmentation = random.choice(['0','90','180','270','H-Flip','V-Flip'])

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

        patch = new_img[dup1_r1:dup1_r2,dup1_c1:dup1_c2,:]
        augmented_patch = augment_patch(patch, augmentation)
        new_img[dup2_r1:dup2_r2,dup2_c1:dup2_c2,:] = augmented_patch

        dup_coord1 = (dup1_r1,dup1_r2,dup1_c1,dup1_c2)
        dup_coord2 = (dup2_r1,dup2_r2,dup2_c1,dup2_c2)

        mask_img[dup2_r1:dup2_r2,dup2_c1:dup2_c2,0] = 1

    return new_img, mask_img

def create_cut_manipulation(img, resize_dim):
    
    img_h, img_w, _ = img.shape
        
    opt = random.choice(['horz', 'vert'])
    if opt=='horz':
            
        first_seq = [i for i in range(5,(img_h//2)-1)]
        sec_seq = [i for i in range((img_h//2)+1,img_h-5)]
            
        if len(first_seq)==0 or len(sec_seq)==0:
            return img, np.zeros_like(img[:,:,:1])
            
        cutoff1 = random.choice(first_seq)
        cutoff2 = random.choice(sec_seq)
        
        img1 = img[:cutoff1,:,:]
        img2 = img[cutoff2:,:,:]
            
        img = np.concatenate((img1, img2), axis=0)
    
        gt = np.zeros_like(img[:,:,:1])
        gt[cutoff1-3:cutoff1+3,:,:] = 1
            
    else:
            
        first_seq = [i for i in range(5,(img_w//2)-1)]
        sec_seq = [i for i in range((img_w//2)+1,img_w-5)]
            
        if len(first_seq)==0 or len(sec_seq)==0:
            return img, np.zeros_like(img[:,:,:1])
            
        cutoff1 = random.choice(first_seq)
        cutoff2 = random.choice(sec_seq)
            
        img1 = img[:,:cutoff1,:]
        img2 = img[:,cutoff2:,:]
            
        img = np.concatenate((img1, img2), axis=1)
            
        gt = np.zeros_like(img[:,:,:1])
        gt[:,cutoff1-3:cutoff1+3,:] = 1
            
    assert img.shape[0]>7 and img.shape[1]>7, '{}'.format(img.shape)
            
    return img, gt

def unmanipulated(img, resize_dim):

    #img = cv2.resize(img, resize_dim)
    gt_mask = np.zeros_like(img[:,:,:1])
    
    return img, gt_mask

def create_manipulation(img, resize_dim):
        
    choices = ['Pristine', 'Cut']#, 'Patch']
    choice = random.choice(choices)

    if choice=='Pristine':
        img, gt_mask = unmanipulated(img, resize_dim)
    elif choice=='Cut':
        img, gt_mask = create_cut_manipulation(img, resize_dim)
    elif choice=='Patch':
        img, gt_mask = create_patch_manipulation(img, resize_dim)
    else:
        print('Invalid choice!')
        raise SystemExit

    img = img[:,:,::-1]

    img = img.astype('float32')/255.*2-1

    return img, gt_mask

def data_generation(list_IDs_temp, batch_size, resize_dim, dim, n_channels):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((batch_size, dim[0], dim[1], n_channels))
    y = np.empty((batch_size, dim[0], dim[1], 1))
        
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        img = cv2.imread(ID)
        if len(img.shape)<3:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=2)
        X, y = create_manipulation(img, resize_dim)

    X = np.expand_dims(X, axis=0)
    y = np.expand_dims(y, axis=0)

    return X, y

def DataGenerator():
    'Generates data for Keras'

    dim = (256,256)
    resize_dim = (256, 256)
    batch_size = 1
    list_IDs = load_image_paths()
    n_channels = 3
    indexes = on_epoch_end(list_IDs)

    while True:

        for index in range(length(list_IDs, batch_size)):
            X, y = get_batch(indexes, index, list_IDs, batch_size, resize_dim, dim, n_channels)
            yield (X,y)

        indexes = on_epoch_end(list_IDs)
