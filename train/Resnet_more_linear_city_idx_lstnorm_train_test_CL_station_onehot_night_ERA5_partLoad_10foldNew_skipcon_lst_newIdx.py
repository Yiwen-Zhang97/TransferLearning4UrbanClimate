#make sure about the cuda version
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms 
import os
import random
import pickle
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import h5py
from dataloaders_city_idx_train_test_ERA5_residual import BuildFineTuneDataLoaders
#from scipy.special import lambertw
import math
import pandas as pd

fold_0=[718, 95, 4793, 296, 3857]
fold_1=[1604, 2534, 289, 638, 55]
fold_2=[116, 305, 346, 258, 412,1008]
fold_3=[1714, 6137, 492, 2118, 4012]
fold_4=[4067, 2487, 3101, 863, 2130]
fold_5=[1998, 2106, 451, 2373, 4023,950]
fold_6=[707, 372, 1777, 3128, 416]
fold_7=[247, 3205, 1519, 2941, 1222]
fold_8=[3783, 2323, 371, 1963,424]
fold_9=[1864, 2484, 3806, 1637, 63]

folder_all = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'fold_6', 'fold_7', 'fold_8', 'fold_9']

# ##########    Change   ###############
fold_val_num = 9
fold_train_tot = 6
dropout_pct=0.1

SCRATCH= False
mode='night'

file_to_store=f'_finetune_1_US_ERA5_{mode}_scratchLastFC_nofreeze_dropout{dropout_pct}_val{fold_val_num}_trainTot{fold_train_tot}_skipcon_lst_2013-2024'
if SCRATCH== False:
    LEARNING_RATE = 5e-4
elif SCRATCH== True:
    LEARNING_RATE = 5e-4

FREEZE = False
RANDOM=True
cities_train=None
cities_test=None
fold_val = 'fold_'+str(fold_val_num)
if fold_train_tot <= (9-fold_val_num):
    fold_train = folder_all[(fold_val_num+1):(fold_val_num+1+fold_train_tot)]
else:
    fold_train = folder_all[(fold_val_num+1):(fold_val_num+1+fold_train_tot)]+folder_all[:(fold_val_num+fold_train_tot-9)]
sites_val = vars()[fold_val]
sites_train = []
for fold in fold_train:
    sites_train.extend(vars()[fold])
sites_test = None
""
PATH_TO_DATA = f'/glade/derecho/scratch/yiwenz/finetune_AllCountries/finetune_city_hdf5_{mode}_1_US_qcMoreLess_0-1_noOutliers_2013-2024/finetune_all.csv'
PATH_TO_MODEL=f"/glade/u/home/yiwenz/TransferLearning/New_Organized_Code/Train_test_results_model/best_resnet_more_linear_samesize_all_cities_LSTNorm_CL_6_L1_onehot_night_ERA5_US_monthly_0-1_all_cont_9.pt"
#PATH_TO_MODEL=f"/glade/u/home/yiwenz/TransferLearning/New_Organized_Code/Train_test_results_model/best_resnet_more_linear_samesize_all_cities_LSTNorm_CL_6_L1_onehot_day_ERA5_US_monthly_26_3_cont_3.pt"
dir_to_store="/glade/u/home/yiwenz/TransferLearning/New_Organized_Code/Train_test_results_model/"


random.seed(42)
train_pct=0.7
EPOCHS = 80
DECAY_RATE = 0.99
DEVICE = "cpu"
BATCH_SIZE = 128


finetune_trainloader, finetune_testloader,len_train_data, len_test_data,valloader,len_val_data = BuildFineTuneDataLoaders(
                                                                    path_to_data_dir = PATH_TO_DATA,
                                                                    batch_size=BATCH_SIZE,
                                                                     train_pct=0.7,
                                                                     random=RANDOM,
                                                                     cities_train=cities_train,
                                                                     cities_test=cities_test,
                                                                     sites_train=sites_train,
                                                                     sites_test=sites_test,
                                                                     sites_val=sites_val)
print(PATH_TO_DATA)
print(f'site_train:{sites_train},len_train:{len_train_data}')
print(f'site_test:{sites_test},len_test:{len_test_data}')
print(f'site_val:{sites_val},len_val:{len_val_data}')


#7 channels: ['Red', 'Green', 'Blue', "NIR", "SWIR1","ndbi","ndvi","elevation"]
image_normalize = transforms.Normalize(
                  mean=[0, 0, 0, 0, 0, 0, 0, 2.5839e+02],
                  std=[1, 1, 1, 1, 1, 1, 1, 3.3506e+02]
)

#forcing: ["strd", "tp", "sp", "rh", "temperature", "uwind","vwind","building height"]
#Added building height in forcing
forcing_mean = torch.from_numpy(np.array([1.1066e+06,  5.6395e-04,  9.8393e+04,  6.8529e+01, 2.8995e+02,  3.8538e-01,  1.6821e-01,  6.2637e+00]))
forcing_std = torch.from_numpy(np.array([2.1263e+05, 2.0910e-03, 4.2614e+03, 1.7899e+01, 9.5041e+00, 2.7570e+00, 3.0576e+00, 2.2420e+00]))

lst_mean = torch.from_numpy(np.array([283.8263]))
lst_std = torch.from_numpy(np.array([9.3311]))

def process_data(forcing, image, month, t2m, lst):
    image, forcing, t2m, lst= image.to(DEVICE).to(torch.float32), forcing.to(DEVICE), t2m.to(DEVICE), lst.to(DEVICE)
    month = month-1
    month = month.to(DEVICE).to(torch.int64)
    # Image Transformations
#    image[:,7,] = torch.clip(image[:,7,], min=0, max=600)
    image[:,:5,] = torch.clip(image[:,:5,], min=0, max=1)
    image[:,5:7,] = torch.clip(image[:,5:7,], min=-1, max=1)
    image = image_normalize(image)
#    image = image_rotate(image)
    # Forcing Transformation
    forcing = forcing[:,1:]
    forcing = torch.div(torch.sub(forcing, forcing_mean), forcing_std).to(torch.float32)
#    forcing = forcing.unsqueeze(2).unsqueeze(3)
#    forcing = forcing.repeat(1,1,33,33)
    # LST Transformation
    lst = torch.div(torch.sub(lst, lst_mean), lst_std).to(torch.float32).view(-1, 1)
    lst = lst.view(-1, 1).to(torch.float32)
    t2m = torch.div(torch.sub(t2m, lst_mean), lst_std).to(torch.float32).view(-1, 1)
    t2m = t2m.view(-1, 1).to(torch.float32)
    return forcing, image, month, t2m, lst

class SuperLoss(nn.Module):

    def __init__(self, C=10, lam=1, batch_size=128):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size
                  
    def forward(self, logits, targets):
        l_i = F.cross_entropy(logits, targets, reduction='none').detach()
        sigma = self.sigma(l_i)
        loss = (F.cross_entropy(logits, targets, reduction='none') - self.tau)*sigma + self.lam*(torch.log(sigma)**2)
        loss = loss.sum()/self.batch_size
        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size())*(-2/math.exp(1.))
        x = x.cuda()
        y = 0.5*torch.max(x, (l_i-self.tau)/self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
#        out = F.dropout2d(out, p=0.3)
        out = self.bn1(out)
        out = F.leaky_relu(out)

        out = self.conv2(out)
#        out = F.dropout2d(out, p=0.1)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.leaky_relu(out)

        return out       

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        in_channel=8,
        forcing_shape = 8,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.flatten_shape = None
        
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool1 = nn.AvgPool2d(2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
#        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))

        self.fc0_1 = nn.Linear(forcing_shape, out_features=64)
        self.fc0_2 = nn.Linear(64+12, out_features=128)
        self.fc0_3 = nn.Linear(128, out_features=256)
        self.fc1 = nn.Linear(128+256, out_features=512)
        self.drop1 = nn.Dropout(dropout_pct)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1)

        self.fc_skip0_1 = nn.Linear(1, out_features=64)
        self.fc_skip0_2 = nn.Linear(64, out_features=128)
        self.fc_skip0_3 = nn.Linear(128, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = nn.ModuleList([])
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _skip_connection(self,lst) -> Tensor:

        lst = self.fc_skip0_1(lst)
        lst = F.leaky_relu(lst)
        lst = self.fc_skip0_2(lst)
        lst = F.leaky_relu(lst)
        lst = self.fc_skip0_3(lst)
        
        return lst

    def forward(self, x: Tensor, forcing, one_hot_mon, lst) -> Tensor:
        # See note [TorchScript super()]
#        x = self.conv0(x)
#        x = x.permute([0,3,1,2])

        lst_s = self._skip_connection(lst)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.avgpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool2(x)
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = F.leaky_relu(x)
        forcing = self.fc0_1(forcing)
        # x = self.drop1(x) #new
        forcing = F.leaky_relu(forcing)
        forcing = torch.cat((forcing, one_hot_mon), dim=1)
        forcing = self.fc0_2(forcing)
        # x = self.drop1(x) #new
        forcing = F.leaky_relu(forcing)
        forcing = self.fc0_3(forcing)
        x = torch.cat((x, forcing), dim=1)     
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)

        x = x+lst_s

        return x


def resnet_simplified():
    return ResNet(BasicBlock,[3,3,0,0])
def resnet_bottleneck():
    return ResNet(Bottleneck,[3,3,0,0])
def se_resnet():
    return ResNet(SEBasicBlock,[2,2,2,2])

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def sigma_li(l_i,tau,lam):
    beta=(l_i-tau)/lam
    sigma=(math.e)**(-beta)
#    y=torch.max(torch.tensor(-2/math.e),beta)
#    sigma=(math.e)**(-1/2*beta)
    return sigma

def SuperLoss(l_i,tau,lam):
    sigma=sigma_li(l_i,tau,lam)
    loss=(l_i-tau)*sigma+lam*(torch.log(sigma))**2
    return loss

def load_pretrained_weights(model,path,freeze=True):
    print(f'Loaded trained weights from {PATH_TO_MODEL}')
    pretrained_dict = OrderedDict()
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        pretrained_dict[name] = v
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc3' not in k} # and 'fc2' not in k and 'fc1' not in k
    print('Param loaded from pretrain:', pretrained_dict.keys())
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    if freeze is True:
        print('param to be updated:')
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'fc3' in name or 'fc2' in name or 'fc1' in name:
                print(name)
                param.requires_grad = True

""
model=resnet_simplified()
if SCRATCH is False:
    load_pretrained_weights(model,PATH_TO_MODEL,freeze=FREEZE)

model = model.to(DEVICE)
model = torch.nn.DataParallel(model)
loss_fn_l1 = nn.L1Loss() #loss_fn = RMSELoss
loss_fn = nn.L1Loss()
#loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

test_loss = []
train_loss = []
test_loss_l1 = []
train_loss_l1 = []
val_loss_l1 = []
tau_init = 0.05

lst_mean = lst_mean.to(DEVICE)
lst_std = lst_std.to(DEVICE)
forcing_mean = forcing_mean.to(DEVICE)
forcing_std = forcing_std.to(DEVICE)

min_test_loss = np.inf 
torch.backends.cudnn.benchmark = True
for epoch in range(EPOCHS):
    print("****** EPOCH: [{}/{}] LR: {} ******".format(epoch, EPOCHS, round(optimizer.param_groups[0]['lr'], 6)))
    running_train_loss = 0
    running_train_l1 = 0 
    train_n_iter = 0
    running_test_loss = 0
    running_test_l1 = 0 
    test_n_iter = 0
    running_val_l1 = 0 
    val_n_iter = 0
    
    if epoch==0:
        tau_running = tau_init
    else:
        tau_running = avg_train_loss
        
    print(f"tau:{tau_running}")

    model.train()
#    loop_train = tqdm(pretrain_trainloader, total=(len_train//BATCH_SIZE) + 1, leave=True)
    for idx, (forcing, image, month, t2m, lst) in enumerate(finetune_trainloader):

        t2m_l1=t2m.view(-1, 1).to(DEVICE).to(torch.float32)
        forcing, image, month, t2m, lst = process_data(forcing, image, month, t2m, lst)
        one_hot_mon = F.one_hot(month, num_classes=12)
        optimizer.zero_grad()
        forward_out = model.forward(image, forcing, one_hot_mon,lst)
        loss = loss_fn(forward_out, t2m)
        super_loss = SuperLoss(loss,tau_running,0.1)
        forward_out_l1=torch.add(torch.multiply(forward_out,lst_std),lst_mean)
        train_l1 = loss_fn_l1(forward_out_l1, t2m_l1)
        super_loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        running_train_l1 += train_l1.item()
        train_n_iter += 1

#    loop_test = tqdm(pretrain_testloader, total=(len_test//BATCH_SIZE) + 1, leave=False)

    model.eval()
    with torch.no_grad():
        for idx, (forcing, image, month, t2m, lst) in enumerate(finetune_testloader):
            t2m_l1=t2m.view(-1, 1).to(DEVICE).to(torch.float32)
            forcing, image, month, t2m, lst = process_data(forcing, image, month, t2m, lst)
            one_hot_mon = F.one_hot(month, num_classes=12)
            pred = model.forward(image, forcing, one_hot_mon,lst)
            testloss = loss_fn(pred, t2m)
            pred_l1=torch.add(torch.multiply(pred,lst_std),lst_mean)
            test_l1 = loss_fn_l1(pred_l1, t2m_l1)
            running_test_loss += testloss.item()
            running_test_l1 += test_l1.item()
            test_n_iter += 1
#            loop_test.set_postfix(test_loss=testloss.item())
        for idx, (forcing, image, month, t2m, lst) in enumerate(valloader):
            t2m_l1=t2m.view(-1, 1).to(DEVICE).to(torch.float32)
            forcing, image, month,t2m, lst = process_data(forcing, image, month, t2m, lst)
            one_hot_mon = F.one_hot(month, num_classes=12)
            pred = model.forward(image, forcing, one_hot_mon,lst)
            pred_l1=torch.add(torch.multiply(pred,lst_std),lst_mean)
            val_l1 = loss_fn_l1(pred_l1, t2m_l1)
            running_val_l1 += val_l1.item()
            val_n_iter += 1

    avg_train_loss = running_train_loss/train_n_iter
    train_loss.append(avg_train_loss)
    avg_test_loss = running_test_loss/test_n_iter
    test_loss.append(avg_test_loss)
    
    avg_train_l1 = running_train_l1/train_n_iter
    avg_test_l1 = running_test_l1/test_n_iter
    train_loss_l1.append(avg_train_l1)
    test_loss_l1.append(avg_test_l1)
    
    avg_val_l1 = running_val_l1/val_n_iter
    
    scheduler.step()
    scheduler2.step(avg_test_loss)
    if avg_test_loss < min_test_loss:
        print("Saving Model")
        min_test_loss = avg_test_loss
        torch.save(model.state_dict(), os.path.join(dir_to_store,"best_resnet"+file_to_store+".pt"))
#    torch.save(model.state_dict(), os.path.join(dir_to_store,f"resnet_{file_to_store}_epoch_{epoch}.pt"))

    print("Saving Loss")
    file_name = os.path.join(dir_to_store,"train_loss_l1"+file_to_store+".pkl")
    open_file = open(file_name, "wb")
    pickle.dump(train_loss_l1, open_file)
    open_file.close()
    file_name = os.path.join(dir_to_store,"test_loss_l1"+file_to_store+".pkl")
    open_file = open(file_name, "wb")
    pickle.dump(test_loss_l1, open_file)
    open_file.close()
    print("------ Train Loss: {}, Test Loss: {} ------".format(avg_train_loss, avg_test_loss))
    print("------ Train L1 Loss: {}, Test L1 Loss: {} ------".format(avg_train_l1, avg_test_l1))
    print("------ Val L1 Loss: {} ------".format(avg_val_l1))
