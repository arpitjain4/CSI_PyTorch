#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:52:04 2023

@author: abhishek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:15:22 2023

@author: abhishek
"""

import os
import time
import torch
import torch.nn as nn
from utils.parser import args
from utils import logger, Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataset import DataLoader38901
#from tensorboardX import SummaryWriter
#from torchviz import make_dot
from models import csinet
#import random
import thop
#data_dir = r"C:\Users\abhis\OneDrive - IIT Kanpur\OnedriveIIT\Programming\AIML_CSI_ENHANCE Programming\dataset\COST2100\dataset"
data_dir = r"/home/abhishek/dataset/38.901/"
model_dir ="/home/abhishek/OneDrive/OnedriveIIT/Programming/trained_models"
reduction=16
#args.cpu = True
device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)
data_type = "uma_rank1_in_los_sgcs"
#PYTORCH_CUDA_ALLOC_CONF ="TRUE"

############ Create the data loader

train_loader, val_loader, test_loader = DataLoader38901(
     root=data_dir,
     batch_size=100,
     num_workers=16,
     pin_memory=pin_memory,
     scenario='in')()


############################## Define model ###########################################
 ##########    Model loading
 #model = dualnet_raw(reduction=4)
model = csinet(reduction)
 #print(model)
 
#H_a = torch.randn([1,2,32,32]).to(device)
#flops, params = thop.profile(model, inputs=(H_a,), verbose=True)
#flops, params = thop.clever_format([flops, params], "%.3f")
#model = init_model(args)
model.to(device)

## load trained model
#model.load_state_dict(torch.load(f'{model_dir}/{data_type}_{reduction}.pt'))
 # Define loss function
criterion_mse = nn.MSELoss().to(device)
#criterion_cs = nn.CosineSimilarity(dim=-1).to(device)
# Inference mode
if args.evaluate:
    loss1, rho1, nmse1 = Tester(model, device, criterion_mse)(test_loader)
    print(f"\n=! Final test loss: {loss1:.3e}"
    f"\n         test rho: {rho1:.3e}"
    f"\n         test NMSE: {nmse1:.3e}\n")
    loss2, rho2, nmse2 = Tester(model, device, criterion_cs)(test_loader)
    print(f"\n=! Final test loss: {loss2}"
    f"\n         test rho: {rho2:.3e}"
    f"\n         test NMSE: {nmse2:.3e}\n")
 

# Define optimizer and scheduler

lr_init = 1e-4 if args.scheduler == 'const' else 2e-3
optimizer = torch.optim.Adam(model.parameters())

if args.scheduler == 'const':
 scheduler = FakeLR(optimizer=optimizer)

else:
 scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                 T_max=args.epochs * len(train_loader),
                                 T_warmup=30 * len(train_loader),
                                 eta_min=5e-5)

# Define the training pipeline

trainer = Trainer(model=model,
           device=device,
           optimizer=optimizer,
           criterion=criterion_mse,
           scheduler=scheduler,
           resume=args.resume)
trainer.loop(args.epochs, train_loader, val_loader, test_loader)

 # Final testing
loss, rho, nmse = Tester(model, device, criterion_mse)(test_loader)
print(f"\n=! Final test loss: {loss:.3e}"
   f"\n         test rho: {rho:.3e}"
   f"\n         test NMSE: {nmse:.3e}\n")

#Save model
torch.save(model.state_dict(), f'{model_dir}/{data_type}_{reduction}.pt')
dummy = torch.randn(1,2,32,32).to(device)
torch.onnx.export(model,dummy,f'{model_dir}/{data_type}_{reduction}.onnx')