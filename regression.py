# %%
import torch
import torch.nn as nn
import numpy as np
import tqdm.auto as tqdm


# the input shape is (batch_size, 5) and the output shape is (batch_size, 1)
class Regression(nn.Module):
    def __init__(self, normalize_constant=None, label_normalize_constant=None):
        super(Regression, self).__init__()
        # self.param_div = nn.Parameter(torch.rand(5).float().cuda())
        self.param_mul = nn.Parameter(torch.rand(5).float().cuda())
        self.linear = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU(),
        )
        self.normalize_constant = normalize_constant
        self.label_normalize_constant = label_normalize_constant
    
    @torch.no_grad()
    def get_normalize_constant(self):
        return self.normalize_constant

    @torch.no_grad()
    def get_label_normalize_constant(self):
        return self.label_normalize_constant
    
    @torch.no_grad()
    def normalized2label(self, x):
        return x * self.label_normalize_constant[1] + self.label_normalize_constant[0]

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
    @torch.no_grad()
    def predict(self, x):
        return self.forward(x)
    

# the Dataset class for regression. The data is from np_batch.npy file and the label is from np_label.npy file
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path):
        self.data = torch.from_numpy(np.load(data_path)).float()
        self.label = torch.from_numpy(np.load(label_path)).float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# the load dataset
train_ds = RegressionDataset('np_batch.npy', 'np_label.npy')

# print the parts of np_batch.npy and np_label.npy

# normalize the data and label
mean = train_ds.data.mean()
std = train_ds.data.std()

label_mean = train_ds.label.mean()
label_std = train_ds.label.std()

train_ds.data = (train_ds.data - train_ds.data.mean()) / train_ds.data.std()
train_ds.label = (train_ds.label - train_ds.label.mean()) / train_ds.label.std()

# the dataloader
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=8)

# the model
model = Regression(normalize_constant=(mean, std), label_normalize_constant=(label_mean, label_std)).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
criterion = nn.MSELoss()
epochs = 1_000

# the training loop with tqdm with set_postfix which contains the loss
progress_bar = tqdm.trange(epochs)
model.train()
for epoch in progress_bar:
    for x, y in train_dl:
        x, y = x.cuda(), y.cuda()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'loss': loss.item(), 'pred': y_pred.mean().item(), 'label': y.mean().item(), 'diff': (y_pred.mean() - y.mean()).item(), 'epoch': epoch})
        

# %%
num_model = 18
num_mix = 32
num_batch = 128
num_worker = 1
num_ga = 1

input_data = torch.tensor([num_model, num_mix, num_batch, num_worker, num_ga]).float().cuda()

# normalize the input data
mean, std = model.get_normalize_constant()
input_data = (input_data - mean) / std

# predict the output
output_data = model.predict(input_data)

# norm2label
output_data = model.normalized2label(output_data)

# print the output
print(output_data.item())

# load np_batch.npy and np_label.npy
batch = np.load('np_batch.npy')
label = np.load('np_label.npy')
print(batch[0], label[0])



# %%
