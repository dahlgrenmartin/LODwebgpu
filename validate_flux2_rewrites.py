#!/usr/bin/env python3
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import torch
from torch._functorch.aot_autograd import aot_export_module

ROOT=Path(__file__).resolve().parent

def load(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

probe=load('probe', ROOT/'flux2_lod_aot_probe.py')
rw=load('rw', ROOT/'flux2_targeted_rewrites.py')

torch.manual_seed(123)
m=probe.JointLOD('sym4').eval()
z=torch.randn(1,32,2,2,requires_grad=True)
target=torch.randn(1,3,16,16)
gm,sig=aot_export_module(m,(z,target),trace_joint=True,output_loss_index=0)
gm2=copy.deepcopy(gm)
stats=rw.rewrite(gm2)
state=m.state_dict()
user_map={sig.user_inputs[0]:z.detach(), sig.user_inputs[1]:target}
args=[]
for n in gm.graph.nodes:
    if n.op!='placeholder': continue
    key=n.name
    if key in sig.inputs_to_parameters: args.append(state[sig.inputs_to_parameters[key]])
    elif key in sig.inputs_to_buffers: args.append(state[sig.inputs_to_buffers[key]])
    elif key in user_map: args.append(user_map[key])
    else: raise KeyError(key)
with torch.no_grad():
    ref=gm(*args); got=gm2(*args)
print('rewrite_stats=',stats)
for i,(a,b) in enumerate(zip(ref,got)):
    if isinstance(a,torch.Tensor):
        print(f'output[{i}] shape={tuple(a.shape)} max_abs_error={(a-b).abs().max().item():.9g} mean_abs_error={(a-b).abs().mean().item():.9g}')
