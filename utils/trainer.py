import torch
from transformers import get_scheduler
from tqdm.auto import tqdm
from .dataset import prepare_dataset
from .model import get_model
from .seed import set_seed
from accelerate import Accelerator
import os
import time
import pickle

# cfg.scheduler
# cfg.scheduler_num_warmup_steps
# cfg.num_epochs

class Trainer(object):
    def __init__(self, cfg):
        self.accelerator = Accelerator(log_with='wandb', gradient_accumulation_steps=cfg.gradient_accumulation_steps)
        self.accelerator.init_trackers(
            project_name='Formal.AI_Framework',
            config=vars(cfg),
            init_kwargs={
                'wandb': {
                    'name': f'{cfg.model}',
                    'tags': cfg.tag,
                }
            }
        )
        self.accelerator.wait_for_everyone()
        self.wandb = self.accelerator.get_tracker('wandb')
        set_seed(cfg)

        if self.accelerator.on_main_process:
            print("#################### cfg ####################")
            print(cfg)

        (self.train_loader, self.val_loader) = prepare_dataset(cfg)
        self.model = get_model(cfg).cuda()
        self.optimizer = torch.optim.Adagrad(self.model.parameters())
        self.scheduler = get_scheduler(
            'linear',
            self.optimizer,
            100,
            5 * len(self.train_loader),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.results = {
            'real_time': {
                'forward_time': [],
                'calc_loss_time': [],
                'backward_time': [],
                'update_time': []
            }, 
            'cuda_time': {
                'forward_time': [],
                'calc_loss_time': [],
                'backward_time': [],
                'update_time': []
            }
        }
        
        self.cfg = cfg
        
        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader
        )
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
    
    def train(self, epoch):
        progress_bar = tqdm(self.train_loader, leave=True)
        self.model.train()

        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                inputs, targets = batch
                
                # forward
                self.starter.record()
                wall_forward_time1 = time.time_ns()
                outputs = self.model(inputs)
                wall_forward_time2 = time.time_ns()
                self.ender.record()
                wall_forward_time = wall_forward_time2 - wall_forward_time1
                torch.cuda.synchronize()
                cuda_forward_time = self.starter.elapsed_time(self.ender)
                
                # calc loss
                self.starter.record()
                wall_calc_time1 = time.time_ns()
                loss = self.loss_fn(outputs, targets)
                wall_calc_time2 = time.time_ns()
                self.ender.record()
                wall_calc_time = wall_calc_time2 - wall_calc_time1
                torch.cuda.synchronize()
                cuda_calc_time = self.starter.elapsed_time(self.ender)
        
                # backward
                self.starter.record()
                wall_backward_time1 = time.time_ns()
                self.accelerator.backward(loss)
                wall_backward_time2 = time.time_ns()
                self.ender.record()
                wall_backward_time = wall_backward_time2 - wall_backward_time1
                torch.cuda.synchronize()
                cuda_backward_time = self.starter.elapsed_time(self.ender)

                # update
                self.starter.record()
                wall_update_time1 = time.time_ns()
                self.optimizer.step()
                self.scheduler.step()
                wall_update_time2 = time.time_ns()
                self.ender.record()
                wall_update_time = wall_update_time2 - wall_update_time1
                torch.cuda.synchronize()
                cuda_update_time = self.starter.elapsed_time(self.ender)
                self.optimizer.zero_grad()

                self.results['real_time']['forward_time'].append(wall_forward_time)
                self.results['real_time']['calc_loss_time'].append(wall_calc_time)
                self.results['real_time']['backward_time'].append(wall_backward_time)
                self.results['real_time']['update_time'].append(wall_update_time)

                self.results['cuda_time']['forward_time'].append(cuda_forward_time)
                self.results['cuda_time']['calc_loss_time'].append(cuda_calc_time)
                self.results['cuda_time']['backward_time'].append(cuda_backward_time)
                self.results['cuda_time']['update_time'].append(cuda_update_time)
   
    def fit(self):
        for epoch in range(5):
            self.train(epoch)
        self.save_results()

    def save_results(self):
        if not os.path.exists('result'):
            os.makedirs('result', exist_ok=True)
        
        file_dir = f'result/{self.cfg.model}_{self.cfg.batch_size}_{self.cfg.num_workers}_{self.cfg.gradient_accumulation_steps}_{self.cfg.mixed_precision}'
        with open(file_dir, 'wb') as f:
            pickle.dump(self.results, f)
            