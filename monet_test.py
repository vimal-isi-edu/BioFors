import torch
from torch import nn
import cv2
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from models.monet import MultiScaleModel
from dataloader.monet_dataloader import MonetTestDL

parser = argparse.ArgumentParser(description='Test trained Monet model on BioFors EDD task.')
parser.add_argument('--model-path', help='path to trained model', default='./model_weights/monet_regular_margin_model.pt', required=True)
parser.add_argument('--image-categories', nargs='+', default=['Blot/Gel', 'Microscopy', 'Macroscopy', 'FACS'], help='Choose one/all of the following categories - Blot/Gel, Microscopy, Macroscopy, FACS', required=True)
parser.add_argument('--patch-map-root', help='path to patch mapping weights', default='./model_weights/', required=False)
parser.add_argument('--output-dir', help='path to output directory', default='./predictions/EDD/', required=False)
parser.add_argument('--annotation-dir', help='path to annotation file directory', default='./annotation_files/', required=False)
parser.add_argument('--image-dir', help='path to image/dataset directory', default='./biofors_images/', required=False)
parser.add_argument('--start-k', help='starting kernel size', default=11, required=False)
parser.add_argument('--batch-size', help='dataloader batch size', default=128, required=False)
parser.add_argument('--device', help='gpu/cpu device', default='cuda:0', required=False)

if __name__ == '__main__':
    
    args = parser.parse_args()

    # Load model
    model = MultiScaleModel(start_k=args.start_k,
                            patch_map_root=args.patch_map_root)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)
    model.eval()

    # Load test dataloader for BioFors
    test_dl = MonetTestDL(annotation_dir=args.annotation_dir,
                          image_dir=args.image_dir,
                          image_categories=args.image_categories)
    test_dset = DataLoader(test_dl, batch_size=args.batch_size, shuffle=False, num_workers=32)

    # Inference loop
    with torch.no_grad():
        print('Predicting masks for samples ...')
        for img1, img2, s1, s2, doi, panel1, panel2 in tqdm(test_dset):
        
            img1 = img1.to(args.device)
            img2 = img2.to(args.device)
        
            sm1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(True, 
                                                                     False, 
                                                                     img1, img2, 
                                                                     None, None, None, None, None)
            sm2, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(True, 
                                                                     False, 
                                                                     img2, img1, 
                                                                     None, None, None, None, None)
        
            sm1 = sm1.to('cpu').numpy()
            sm2 = sm2.to('cpu').numpy()
        
            # interpolate
            samples = img1.shape[0]
            for i in range(samples):
            
                mask1 = cv2.resize(sm1[i], (s1[i][1], s1[i][0]))
                mask2 = cv2.resize(sm2[i], (s2[i][1], s2[i][0]))
            
                mask1[mask1>0.] = 1
                mask1[mask1<=0.] = 0
                mask2[mask2>0.] = 1
                mask2[mask2<=0.] = 0
            
                if not os.path.exists(os.path.join(args.output_dir, doi[i])):
                    os.mkdir(os.path.join(args.output_dir, doi[i]))
                
                dir_name = panel1[i] + '__' + panel2[i]
                os.mkdir(os.path.join(args.output_dir, doi[i], dir_name))
            
                cv2.imwrite(os.path.join(args.output_dir, doi[i], dir_name, panel1[i]), mask1)
                cv2.imwrite(os.path.join(args.output_dir, doi[i], dir_name, panel2[i]), mask2)
