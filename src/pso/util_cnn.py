# -----------------------------------
# Author
#   Lorenzo Tronchin
# Module description
#   Module to define the architecture and training procedure of shallow cnn
# Date
#
# -----------------------------------

import time
import os
import pandas as pd
import numpy as np

import copy
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def get_cnn(model_name, image_channels, iid_classes, n_class, img_dim=64, cnn_args=None):
    if model_name == "AlexNet":
        return AlexNet(image_channels, iid_classes,  n_class, img_dim, cnn_args)
    elif model_name == "VGG16":
        pass
    elif model_name == "InceptionNet":
        pass
    elif model_name == "ResNet50":
        return Resnet(resnet_block, [3, 4, 6, 3], image_channels, iid_classes, n_class)
    elif model_name == "ResNet101":
        return Resnet(resnet_block, [3, 4, 23, 3], image_channels, iid_classes, n_class)
    elif model_name == "ResNet152":
        return Resnet(resnet_block, [3, 8, 36, 3], image_channels, iid_classes, n_class)
    else:
        raise ValueError(model_name)


def test(model_name):
    net = get_cnn(model_name=model_name, image_channels=3, img_dim=64)
    x = torch.randn(2, 3, 64, 64)
    y = net(x).to('cpu')
    print(y.shape)


def get_activation(activation_name, alpha=0.2):
    if activation_name == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=alpha)
    elif activation_name == 'ReLU':
        return nn.ReLU()
    else:
        raise ValueError(activation_name)

def get_opti(model_parameters, name, lr, weight_decay=0, beta1=0.5, beta2=0.999, epsilon=0.00000001):
    if name == 'Adam':
        return optim.Adam(params=model_parameters, lr=lr, betas=(beta1, beta2),  eps=epsilon, weight_decay=weight_decay)
    elif name == 'RMSprop':
        return optim.RMSprop(params=model_parameters, lr=lr, eps=epsilon, weight_decay=weight_decay)
    else:
        raise ValueError(name)


def get_initializer(initializer_name, w):
    if initializer_name == 'glorot_normal':
        return nn.init.xavier_normal_(tensor=w, gain=1.0)
    elif initializer_name == 'he_normal':
        return nn.init.kaiming_normal_(tensor=w, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif initializer_name == 'random_normal':
        return nn.init.normal_(tensor=w, mean=0.0, std=0.02)
    else:
        raise ValueError(initializer_name)


def initialize_weights(model, initializer_name):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            get_initializer(initializer_name, m.weight.data)

class Resnet(nn.Module): # layers --> is a list defining the number of calls to resnet_block (for ResNet50 is [3, 4, 6, 3])
    def __init__(self, resnet_block, layers, image_channels, iid_classes, n_class=2):
        super(Resnet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) # initial layer of resnet
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.class_to_idx = {c: i for i, c in enumerate(sorted(iid_classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # ResNet layers
        self.layer1 = self._make_layer(resnet_block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(resnet_block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(resnet_block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(resnet_block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(512*4, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def forward_avgpool(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        return x

    def _make_layer(self, resnet_block, num_residual_blocks, intermediate_channels, stride):
        """ Function to create a ResNet convx_x layer
            Args:
                resnet_block: class that defines one conv block,
                num_residual_blocks: number of times the resnet_block is called,
                intermediate_channels: number of channels outputted from the resnet_block
                stride: stride of the conv block
              """
        identity_downsample = None
        layers = []
        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes we need to adapt the Identity
        # (skip connection) so it will be able to be added to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels * 4)
            )
        layers.append(resnet_block(self.in_channels, intermediate_channels, identity_downsample, stride)) # this is the layer that changes the output number of channels
        self.in_channels = intermediate_channels * 4 # new input channels
        for i in range(num_residual_blocks - 1):
            layers.append(resnet_block(self.in_channels, intermediate_channels, identity_downsample=None, stride=1))
        return nn.Sequential(*layers)



class resnet_block(nn.Module): # single block that we reuse multiple times to build the ResNet architecture
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(resnet_block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False) # reduce the dimension
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone() # return a copy of x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, image_channels, iid_classes, n_class, img_size, cnn_args):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=cnn_args['kernel'], padding=cnn_args['padding'])
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=cnn_args['kernel'], padding=cnn_args['padding'])
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=cnn_args['kernel'], padding=cnn_args['padding'])
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=cnn_args['kernel'], padding=cnn_args['padding'])
        self.max_pool = nn.MaxPool2d((2,2))
        self.act = get_activation(cnn_args['cnn_activation'])

        self.class_to_idx = {c: i for i, c in enumerate(sorted(iid_classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        x = torch.randn(img_size, img_size).view(-1, 1, img_size, img_size) # create random data
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3= nn.Linear(256, n_class)
        self.dropout = nn.Dropout2d(0.5)

    def convs(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.act(x)
        x = self.max_pool(x)

        if self._to_linear is None:  # if we have not yet calculated what it takes to flatten (self._to_linear), we want to do that.
            self._to_linear = x.shape[1] * x.shape[2] * x.shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)

        #x = x.view(-1, self._to_linear)
        x = x.contiguous().view(-1, self._to_linear) # https://discuss.pytorch.org/t/view-size-is-not-compatible-with-input-tensors-size-and-stride/121488
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 13 * 13, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.act_output(x)
        return x

def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs, early_stopping, model_dir, device, label=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [],
               'train_prec': [], 'val_prec': [], 'train_rec': [], 'val_rec': [],}
    epochs_no_improve = 0
    early_stop = False
    best_epoch = num_epochs

    # Iterate over epochs
    for epoch in range(num_epochs):
        print("\nStart of epoch %d" % (epoch))

        for phase in ['train', 'val']: # Each epoch has a training and validation phase
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_f1_score = []
            running_precision = []
            running_recall = []
            lr_state = optimizer.state_dict()['param_groups'][0]['lr']
            #with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}, lr {lr_state}', unit='img') as pbar:
            for x_batch, y_batch in data_loaders[phase]: # Iterate over the batches of the dataset
                x_batch = x_batch.to(device)

                if label is not None:
                    y_batch = y_batch == label
                    y_batch = y_batch.to(torch.uint8)
                y_batch = y_batch.to(device)

                optimizer.zero_grad() # zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'): # track history if only in train
                    output = model(x_batch.float())
                    loss = criterion(output.float(), y_batch)
                    _, preds = torch.max(output, 1)
                    #pbar.set_postfix(**{'loss (batch)': loss.item()})

                    if phase == 'train':  # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * x_batch.size(0)
                running_corrects += torch.sum(preds == y_batch)
                running_f1_score.append(f1_score(y_batch.detach().cpu().numpy(), preds.detach().cpu().numpy()))
                running_precision.append(precision_score(y_batch.detach().cpu().numpy(), preds.detach().cpu().numpy()))
                running_recall.append(recall_score(y_batch.detach().cpu().numpy(), preds.detach().cpu().numpy()))
                #pbar.update(x_batch.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects / len(data_loaders[phase].dataset)
            epoch_f1_score = np.mean(running_f1_score)
            epoch_precision = np.mean(running_precision)
            epoch_recall = np.mean(running_recall)

            if phase == 'val':
                scheduler.step(epoch_loss)

            if phase == 'train': # update history
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.detach().cpu().numpy().item())
                history['train_f1'].append(epoch_f1_score)
                history['train_prec'].append(epoch_precision)
                history['train_rec'].append(epoch_recall)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.detach().cpu().numpy().item())
                history['val_f1'].append(epoch_f1_score)
                history['val_prec'].append(epoch_precision)
                history['val_rec'].append(epoch_recall)

            print('{} Loss: {:.4f} Acc: {:.4f} f1_score: {:.4f} Prec: {:.4f} Rec: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1_score, epoch_precision, epoch_recall))

            # Early stopping
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping: # Trigger early stopping
                        print(f'\nEarly Stopping! Total epochs: {epoch}%')
                        early_stop = True
                        break
        if early_stop:
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # Save model
    if label is not None:
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_{label}.pt"))
    else:
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()
    return model, history

def train_model_multipatient(model, data_loaders, criterion, optimizer, scheduler, num_epochs, early_stopping, model_dir, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [],
               'train_prec': [], 'val_prec': [], 'train_rec': [], 'val_rec': [],}
    epochs_no_improve = 0
    early_stop = False
    best_epoch = num_epochs

    # Iterate over epochs
    for epoch in range(num_epochs):
        print("\nStart of epoch %d" % (epoch,))

        for phase in ['train', 'val']: # Each epoch has a training and validation phase
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_f1_score = []
            running_precision = []
            running_recall = []
            lr_state = optimizer.state_dict()['param_groups'][0]['lr']
            #with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}, lr {lr_state}', unit='img') as pbar:
            for x_batch, y_batch in data_loaders[phase]: # Iterate over the batches of the dataset
                x_batch = x_batch.to(device)
                y_batch = torch.tensor([model.class_to_idx[y.item()] for y in y_batch], dtype=torch.uint8, device=device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad() # zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'): # track history if only in train
                    output = model(x_batch.float()) # output = model.forward_2(x_batch.float())
                    loss = criterion(output.float(), y_batch)
                    _, preds = torch.max(output, 1)
                    #pbar.set_postfix(**{'loss (batch)': loss.item()})

                    if phase == 'train':  # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * x_batch.size(0)
                running_corrects += torch.sum(preds == y_batch)
                running_f1_score.append(f1_score(y_batch.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro'))
                running_precision.append(precision_score(y_batch.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro'))
                running_recall.append(recall_score(y_batch.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro'))
                #pbar.update(x_batch.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects / len(data_loaders[phase].dataset)
            epoch_f1_score = np.mean(running_f1_score)
            epoch_precision = np.mean(running_precision)
            epoch_recall = np.mean(running_recall)

            if phase == 'val':
                scheduler.step(epoch_loss)

            if phase == 'train': # update history
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.detach().cpu().numpy().item())
                history['train_f1'].append(epoch_f1_score)
                history['train_prec'].append(epoch_precision)
                history['train_rec'].append(epoch_recall)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.detach().cpu().numpy().item())
                history['val_f1'].append(epoch_f1_score)
                history['val_prec'].append(epoch_precision)
                history['val_rec'].append(epoch_recall)

            print('{} Loss: {:.4f} Acc: {:.4f} f1_score: {:.4f} Prec: {:.4f} Rec: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1_score, epoch_precision, epoch_recall))

            # Early stopping
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping: # Trigger early stopping
                        print(f'\nEarly Stopping! Total epochs: {epoch}%')
                        early_stop = True
                        break
        if early_stop:
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # Save model
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()
    return model, history