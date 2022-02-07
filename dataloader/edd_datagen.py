import numpy as np
import keras
import json
from tqdm import tqdm
import cv2
import random
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as keras_image

def length(list_IDs, batch_size):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(list_IDs) / batch_size))

def load_image_paths(img_categories):

    with open('../from_scratch_train_pairs.json', 'r') as f:
        pair_dict = json.load(f)

    list_IDs = []
    for doi in tqdm(pair_dict.keys()):
        for category in img_categories:
            if category in pair_dict[doi]:
                for pair in pair_dict[doi][category]:
                    img1, img2 = pair.split()
                    path1 = '/nas/medifor/esabir/scientific_integrity/from_scratch/from_scratch_panels/'+doi+'/'+img1
                    path2 = '/nas/medifor/esabir/scientific_integrity/from_scratch/from_scratch_panels/'+doi+'/'+img2
                    list_IDs += [(path1, path2)]
                    
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
    X1, X2, y1, y2 = data_generation(list_IDs_temp, batch_size, resize_dim, dim, n_channels, augmentation_list)

    return X1, X2, y1, y2

def create_spliced_manipulation(img1, img2, resize_dim, augmentation_list):
    
    img1 = cv2.resize(img1, resize_dim)
    img2 = cv2.resize(img2, resize_dim)
    
    gt_mask1 = np.zeros_like(img1[:,:,0])
    gt_mask2 = np.zeros_like(img2[:,:,0])
    
    h1,w1,_ = img1.shape
    h2,w2,_ = img2.shape

    size_th = 0.3 # patches atleast 30% in size
    
    r1,c1 = random.randint(0,int(np.floor((1-size_th)*h1))), random.randint(0,int(np.floor((1-size_th)*w1)))
    r2,c2 = random.randint(r1+int(np.floor(size_th*h1)),h1), random.randint(c1+int(np.floor(size_th*w1)),w1)
    
    patch1 = img1[r1:r2,c1:c2,:]
    gt_mask1[r1:r2,c1:c2] = 1
        
    augmentation = random.choice(augmentation_list)
    if augmentation=='H-Flip':
        patch1 = np.fliplr(patch1)
    elif augmentation=='V-Flip':
        patch1 = np.flipud(patch1)
    elif augmentation=='90':
        patch1 = np.rot90(patch1, 1)
    elif augmentation=='180':
        patch1 = np.rot90(patch1, 2)
    elif augmentation=='270':
        patch1 = np.rot90(patch1, 3)
        
    ph, pw, _ = patch1.shape
    
    start_r1, start_c1 = random.randint(0, h1-ph), random.randint(0, w1-pw)
    img2[start_r1:start_r1+ph, start_c1:start_c1+pw, :] = patch1
    gt_mask2[start_r1:start_r1+ph, start_c1:start_c1+pw] = 1

    return img1, img2, gt_mask1, gt_mask2

def create_overlap_manipulation(img1, resize_dim, augmentation_list):

    h1, w1, _ = img1.shape

    gt_mask = np.zeros_like(img1[:,:,0])
    
    r1, c1 = random.randint(0, h1-10), random.randint(0,w1-10)
    r2, c2 = random.randint(r1+5, h1), random.randint(c1+5, w1)
    
    patch1 = img1[r1:r2,c1:c2,:]
    ph, pw, _ = patch1.shape

    h_mid = int(np.floor((r1+min(r2,h1-ph))/2))
    w_mid = int(np.floor((c1+min(c2,w1-pw))/2))

    start_r1, start_c1 = random.randint(r1, h_mid), random.randint(c1, w_mid)
    patch2 = img1[start_r1:start_r1+ph, start_c1:start_c1+pw, :]

    gt_mask[start_r1:r2, start_c1:c2] = 1
    gt_mask1 = gt_mask[r1:r2,c1:c2]
    gt_mask2 = gt_mask[start_r1:start_r1+ph, start_c1:start_c1+pw]

    augmentation = random.choice(augmentation_list)
    if augmentation=='H-Flip':
        patch2 = np.fliplr(patch2)
        gt_mask2 = np.fliplr(gt_mask2)
    elif augmentation=='V-Flip':
        patch2 = np.flipud(patch2)
        gt_mask2 = np.flipud(gt_mask2)
    elif augmentation=='90':
        patch2 = np.rot90(patch2, 1)
        gt_mask2 = np.rot90(gt_mask2, 1)
    elif augmentation=='180':
        patch2 = np.rot90(patch2, 2)
        gt_mask2 = np.rot90(gt_mask2, 2)
    elif augmentation=='270':
        patch2 = np.rot90(patch2, 3)
        gt_mask2 = np.rot90(gt_mask2, 3)
        
    img1 = cv2.resize(patch1, resize_dim)
    img2 = cv2.resize(patch2, resize_dim)

    gt_mask1 = cv2.resize(gt_mask1, resize_dim)
    gt_mask2 = cv2.resize(gt_mask2, resize_dim)

    gt_mask1[gt_mask1>0] = 1
    gt_mask2[gt_mask2>0] = 1
        
    return img1, img2, gt_mask1, gt_mask2

def unmanipulated(img1, img2, resize_dim):
    
    img1 = cv2.resize(img1, resize_dim)
    img2 = cv2.resize(img2, resize_dim)
    
    gt_mask1 = np.zeros_like(img1[:,:,0])
    gt_mask2 = np.zeros_like(img2[:,:,0])
    
    return img1, img2, gt_mask1, gt_mask2

def create_manipulation(img1, img2, resize_dim, augmentation_list):
        
    choices = ['Pristine', 'Splice', 'Overlap']
    choice = random.choice(choices)

    prob = np.zeros((1,2))

    if choice=='Pristine':
        img1, img2, gt_mask1, gt_mask2 = unmanipulated(img1, img2, resize_dim)
        prob[0,1] = 1
    elif choice=='Splice':
        img1, img2, gt_mask1, gt_mask2 = create_spliced_manipulation(img1, img2, resize_dim, augmentation_list)
        prob[0,0] = 1
    elif choice=='Overlap':
        img1, img2, gt_mask1, gt_mask2 = create_overlap_manipulation(img1, resize_dim, augmentation_list)
        prob[0,0] = 1
    else:
        print('Invalid choice!')
        raise SystemExit

    gt_mask = np.concatenate([np.expand_dims(gt_mask1, axis=0),
                              np.expand_dims(gt_mask2, axis=0)], axis=0)

    img1 = img1[:,:,::-1]
    img1 = img1 - np.mean(np.mean(img1,axis=0,keepdims=True),axis=1,keepdims=True)
    img1 = keras_image.img_to_array(img1)
    img1 = preprocess_input(np.expand_dims(img1, axis=0))[0]
    img2 = img2[:,:,::-1]
    img2 = img2 - np.mean(np.mean(img2,axis=0,keepdims=True),axis=1,keepdims=True)
    img2 = keras_image.img_to_array(img2)
    img2 = preprocess_input(np.expand_dims(img2, axis=0))[0]

    return img1, img2, gt_mask, prob

def data_generation(list_IDs_temp, batch_size, resize_dim, dim, n_channels, augmentation_list):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X1 = np.empty((batch_size, n_channels, dim[0], dim[1]))
    X2 = np.empty((batch_size, n_channels, dim[0], dim[1]))
    y1 = np.empty((batch_size, 2, dim[0], dim[1]))
    y2 = np.empty((batch_size, 2))
        
    # Generate data
    for i, (ID1, ID2) in enumerate(list_IDs_temp):
        # Store sample
        img1 = cv2.imread(ID1)
        img2 = cv2.imread(ID2)
        X1[i], X2[i], y1[i], y2[i] = create_manipulation(img1, img2, resize_dim, augmentation_list)

    return X1, X2, y1, y2

def DataGenerator(image_categories, augmentation_list):
    'Generates data for Keras'

    dim = (256,256)
    resize_dim = (256, 256)
    batch_size = 16
    list_IDs = load_image_paths(image_categories)
    n_channels = 3
    indexes = on_epoch_end(list_IDs)

    while True:

        for index in range(length(list_IDs, batch_size)):
            X1, X2, y1, y2 = get_batch(indexes, index, list_IDs, batch_size, resize_dim, dim, n_channels, augmentation_list)
            yield ({'world':X1, 'probe':X2}, {'pred_masks':y1, 'pred_probs':y2})

        indexes = on_epoch_end(list_IDs)
