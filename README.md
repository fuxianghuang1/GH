# Gradient Harmonization in Unsupervised Domain Adaptation

## News
- **[2024-08-02]** All code has been updated.
  
- **[2024-08-02]** The [paper](https://arxiv.org/abs/2408.00288) has been released.

- **[2024-08-01]** The paper "Gradient Harmonization in Unsupervised Domain Adaptation" has been accepted by IEEE TPAMI 2024.

If you find this repository useful, please cite our paper:

```bibtex
@article{huang2024gh,
  title={Gradient Harmonization in Unsupervised Domain Adaptation},
  author={Huang, Fuxiang and Song, Suqi and Zhang, Lei},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2024},
  publisher={IEEE}
}
```

## Prerequisites
- Python: `==3.8`
- PyTorch: `==1.8.1`
- CUDA Toolkit: `==11.1`

## Data Preparation
Download the datasets and extract them to `./data`:
- [Office31](https://faculty.cc.gatech.edu/~judy/domainadapt/)
- [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
- [Visda2017](https://ai.bu.edu/visda-2017/)
- [DomainNet](https://ai.bu.edu/M3SDA/)
- [Digits](https://github.com/thuml/CDAN)

## Training

### Example: Train Baseline + GH

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

### Example: Train Baseline + GH++

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

## Acknowledgements
- **CDAN**: [GitHub Repository](https://github.com/cuishuhao/GVB)
- **MCD**: [GitHub Repository](https://github.com/thuml/Transfer-Learning-Library)
- **DWL**: [GitHub Repository](https://github.com/NiXiao-cqu/TransferLearning-dwl-cvpr2021)
- **GVB**: [GitHub Repository](https://github.com/cuishuhao/GVB)
- **SSRT**: [GitHub Repository](https://github.com/tsun/SSRT)
