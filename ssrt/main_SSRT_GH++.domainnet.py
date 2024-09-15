# Prerequisites for using Huawei Ascend 910: Python==3.8, Pytorch==1.8.1, Ascend-cann-toolkit==6.0.1
# Uncomment the following two lines when running PyTorch on the Ascend AI processor
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

from trainer.train import train_main
import time
import socket
import os

timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
hostName = socket.gethostname()
pid = os.getpid()

domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

for src in domains:
    for tgt in domains:

        if src == tgt:
            continue

        header = '''
        ++++++++++++++++++++++++++++++++++++++++++++++++       
        {}       
        ++++++++++++++++++++++++++++++++++++++++++++++++   
        @{}:{}
        '''.format

        args = ['--model=SSRT',
                '--base_net=vit_base_patch16_224',

                '--gpu=0',
                '--timestamp={}'.format(timestamp),

                '--dataset=DomainNet',
                '--source_path=data/{}_train.txt'.format(src),
                '--target_path=data/{}_train.txt'.format(tgt),
                '--test_path=data/{}_test.txt'.format(tgt),
                '--batch_size=32',

                '--lr=0.004',
                '--train_epoch=40',
                '--save_epoch=40',
                '--eval_epoch=5',
                '--iters_per_epoch=1000',

                '--sr_loss_weight=0.2',
                '--sr_alpha=0.3',
                '--sr_layers=[0,4,8]',
                '--sr_epsilon=0.4',

                '--use_safe_training=True',
                '--adap_adjust_T=1000',
                '--adap_adjust_L=4',

                '--use_tensorboard=False',
                '--tensorboard_dir=tbs/SSRT',
                '--use_file_logger=True',
                '--log_dir=logs/SSRT',

                '--GH=False',
                '--GH_new=True']
        
        train_main(args, header('\n\t\t'.join(args), hostName, pid))