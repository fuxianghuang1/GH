# [IEEE TPAMI 2024] GH/GH++: Gradient Harmonization in Unsupervised Domain Adaptation

The paper ["Gradient Harmonization in Unsupervised Domain Adaptation"](https://arxiv.org/abs/2408.00288) has been accepted by IEEE TPAMI 2024.

## Contents

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training UDA model](#training-uda-model)
- [Training Retrieval model](#training-retrieval-model)
- [Training Detection model](#training-detection-model)
- [Acknowledgements](#acknowledgements)

If you find this repository useful, please consider citing our paper:

```bibtex
@article{huang2024gh,
  title={Gradient Harmonization in Unsupervised Domain Adaptation},
  author={Huang, Fuxiang and Song, Suqi and Zhang, Lei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

## Prerequisites
To run the code, please ensure you have the following dependencies installed:
- Python: `==3.8`
- PyTorch: `==1.8.1`
- CUDA Toolkit: `==11.1`


## Data Preparation
Download the datasets and extract them to `./data`:
- [Office31](https://faculty.cc.gatech.edu/~judy/domainadapt/)
- [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
- [Visda2017](https://ai.bu.edu/visda-2017/)
- [DomainNet](https://ai.bu.edu/M3SDA/)
- [Digits](https://drive.google.com/file/d/1ZUMdVRyXL6EOICCgHRQZXZhrui-8kYvy/view?usp=sharing)
- [CSS](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view)
- [VOC](http://host.robots.ox.ac.uk/pascal/VOC)

## Training UDA model



## Training

### Example: Training Baseline with Gradient Harmonization (GH)

#### CDAN
```bash
python cdan/train_image.py CDAN+E --gpu_id 0 --num_iterations 8004 --dset office --s_dset_path data/office31/amazon.txt --t_dset_path data/office31/webcam.txt --test_interval 500 --output_dir cdan/logs/cdan_gh/office31_a2w --GH True
```

#### MCD
```bash
python mcd/mcd.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 0 -i 500 --trade-off 10.0 --log mcd/logs/mcd_gh/office31_a2w --GH True
```

#### DWL
```bash
python dwl/main.py data/office31 -d Office31 -s A -t W -a resnet50 --GH True
```

#### GVB
```bash
python gvb/train_image.py --gpu_id 0 --GVBG 1 --GVBD 1 --num_iterations 8004 --dset office --s_dset_path data/office31/amazon.txt --t_dset_path data/office31/webcam.txt --test_interval 500 --output_dir gvb/logs/gvb_gh/office31_a2w --GH True
```

#### SSRT
```bash
python ssrt/main_SSRT_GH.office31.py
```

### Example: Training Baseline with Enhanced Gradient Harmonization (GH++)

#### CDAN
```bash
python cdan/train_image.py CDAN+E --gpu_id 0 --num_iterations 8004 --dset office --s_dset_path data/office31/amazon.txt --t_dset_path data/office31/webcam.txt --test_interval 500 --output_dir cdan/logs/cdan_gh/office31_a2w --GH_new True
```

#### MCD
```bash
python mcd/mcd.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 0 -i 500 --trade-off 10.0 --log mcd/logs/mcd_gh/office31_a2w --GH_new True
```

#### DWL
```bash
python dwl/main.py data/office31 -d Office31 -s A -t W -a resnet50 --GH_new True
```

#### GVB
```bash
python gvb/train_image.py --gpu_id 0 --GVBG 1 --GVBD 1 --num_iterations 8004 --dset office --s_dset_path data/office31/amazon.txt --t_dset_path data/office31/webcam.txt --test_interval 500 --output_dir gvb/logs/gvb_gh/office31_a2w --GH_new True
```

#### SSRT
```bash
python ssrt/main_SSRT_GH++.office31.py
```

## Training Retrieval model
### Example: Training Baseline with Enhanced Gradient Harmonization (GH++)
```bash
python Retrieval/maincss_convharmonic.py --GH_new True
```

## Training Detection model
### Example: Training Baseline with Enhanced Gradient Harmonization (GH++)

```bash
python Detection/train.py
```

## Acknowledgements
We would like to acknowledge the following repositories that contributed to our work:
- **CDAN**: [GitHub Repository](https://github.com/cuishuhao/GVB)
- **MCD**: [GitHub Repository](https://github.com/thuml/Transfer-Learning-Library)
- **DWL**: [GitHub Repository](https://github.com/NiXiao-cqu/TransferLearning-dwl-cvpr2021)
- **GVB**: [GitHub Repository](https://github.com/cuishuhao/GVB)
- **SSRT**: [GitHub Repository](https://github.com/tsun/SSRT)
- **GA(TIRG)**: [GitHub Repository](https://github.com/fuxianghuang1/GA)



[NOTE] **If you have any questions, please don't hesitate to contact [Fuxiang Huang](mailto:fxhuang1995@gmail.com), [Suqi Song](mailto:songsuqi@stu.cqu.edu.cn) and [Lei Zhang](mailto:leizhang@cqu.edu.cn).** 
