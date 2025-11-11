import os
import torch
import logging
import torch.distributed as dist
import json
import math
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import LambdaLR


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, ema, scaler, device, is_main_process):
    

    if not checkpoint_path:
        return 0, 0


    if not os.path.exists(checkpoint_path):
        if is_main_process:
            logging.warning(f"Checkpoint path {checkpoint_path} not found. Starting training from scratch.")
        return 0, 0
    
    try:
        if is_main_process:
            logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(filtered_state_dict)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        

        start_iteration = checkpoint['iteration'] + 1
        start_epoch = checkpoint.get('epoch', 0) 

        if is_main_process:
            logging.info(f"Resumed successfully from iteration {start_iteration - 1}.")
            
        return start_iteration, start_epoch

    except Exception as e:
        if is_main_process:
            logging.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            logging.error("Could not load checkpoint. Starting training from scratch.")
        return 0, 0
    
    
def setup_distributed(gpu_ids, backend='nccl'):
    
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return 0, 1, 0 

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    
    local_rank = int(os.environ['LOCAL_RANK'])

    
    if len(gpu_ids) != world_size:
        raise ValueError("The number of specified GPU IDs must match the number of processes (nproc_per_node).")
    
    device_id = gpu_ids[local_rank]
    torch.cuda.set_device(device_id)

    return rank, world_size, device_id


def cleanup_distributed():

    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

class ExponentialMovingAverage_EMA:
    
    
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        model_to_update = model.module if isinstance(model, DDP) else model
        for name, param in model_to_update.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        model_to_update = model.module if isinstance(model, DDP) else model
        for name, param in model_to_update.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
                
                
                
    def restore(self, model):
        model_to_update = model.module if isinstance(model, DDP) else model
        for name, param in model_to_update.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
                
                
    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}
    
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']
                
                