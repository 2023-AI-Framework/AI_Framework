# %%
import os
import pickle
import pandas as pd

data = {
    'model': [],
    
    'real_forwad_time': [],
    'real_calc_loss_time': [],
    'real_backward_time': [],
    'real_update_time': [],
    
    'cuda_forward_time': [],
    'cuda_calc_loss_time': [],
    'cuda_backward_time': [],
    'cuda_update_time': []
}

model = 'resnet101'

skip = 100
for dir in sorted(os.listdir(f'result/{model}')):
    dir = os.path.join(f"result/{model}", dir)
    with open(dir, "rb") as f:
        result = pickle.load(f)
    dir = dir.replace(f"result/{model}", "")
    data['model'].append(dir)
    real_forward_time = result['real_time']['forward_time'][skip:]
    real_calc_loss_time = result['real_time']['calc_loss_time'][skip:]
    real_backward_time = result['real_time']['backward_time'][skip:]
    real_update_time = result['real_time']['update_time'][skip:]
    
    cuda_forward_time = result['cuda_time']['forward_time'][skip:]
    cuda_calc_loss_time = result['cuda_time']['calc_loss_time'][skip:]
    cuda_backward_time = result['cuda_time']['backward_time'][skip:]
    cuda_update_time = result['cuda_time']['update_time'][skip:]
    
    print(dir, len(real_forward_time))
    real_forward_time = sum(real_forward_time) / len(real_forward_time)
    real_calc_loss_time = sum(real_calc_loss_time) / len(real_calc_loss_time)
    real_backward_time = sum(real_backward_time) / len(real_backward_time)
    real_update_time = sum(real_update_time) / len(real_update_time)
    
    cuda_forward_time = sum(cuda_forward_time) / len(cuda_forward_time)
    cuda_calc_loss_time = sum(cuda_calc_loss_time) / len(cuda_calc_loss_time)
    cuda_backward_time = sum(cuda_backward_time) / len(cuda_backward_time)
    cuda_update_time = sum(cuda_update_time) / len(cuda_update_time)
    
    data['real_forwad_time'].append(real_forward_time)
    data['real_calc_loss_time'].append(real_calc_loss_time)
    data['real_backward_time'].append(real_backward_time)
    data['real_update_time'].append(real_update_time)
    
    data['cuda_forward_time'].append(cuda_forward_time)
    data['cuda_calc_loss_time'].append(cuda_calc_loss_time)
    data['cuda_backward_time'].append(cuda_backward_time)
    data['cuda_update_time'].append(cuda_update_time)

df = pd.DataFrame(data)
df.to_csv(f'{model}_no.csv')