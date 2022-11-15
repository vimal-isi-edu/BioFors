# BioFors
Repository for BioFors (Biomedical Image Forensics) Dataset and MONet (Model for duplication detection in biomedical images)

Download [BioFors](https://drive.google.com/file/d/1UVSJ6h7r8pmOWYZkqWeAZ_YvwbFr1wV3/view?usp=sharing).

Sample commands:

Run inference using MONet pretrained model on BioFors test set

```
python monet_test.py --model-path model_weights/monet_regular_margin_model.pt --image-categories FACS Macroscopy
```

Evaluate predictions on EDD task

```
python eval/eval_EDD.py --image-categories FACS Macroscopy
```

If you found out work useful, please cite:

```
@inproceedings{sabir2021biofors,
  title={Biofors: A large biomedical image forensics dataset},
  author={Sabir, Ekraam and Nandi, Soumyaroop and Abd-Almageed, Wael and Natarajan, Prem},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10963--10973},
  year={2021}
}

@inproceedings{sabir2022monet,
  title={MONet: Multi-Scale Overlap Network for Duplication Detection in Biomedical Images},
  author={Sabir, Ekraam and Nandi, Soumyaroop and AbdAlmageed, Wael and Natarajan, Prem},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={3793--3797},
  year={2022},
  organization={IEEE}
}
```