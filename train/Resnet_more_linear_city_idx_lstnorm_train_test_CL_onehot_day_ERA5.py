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
from dataloaders_city_idx_train_test_ERA5 import BuildPreTrainDataLoaders, BuildFineTuneDataLoaders
#from scipy.special import lambertw
import math

PATH_TO_DATA = "/glade/derecho/scratch/yiwenz/pretrain_Shuffled/pretrain_city_hdf5_day_926_ERA5_US_0-1_all_monthly_shuffle"
dir_to_store="/glade/u/home/yiwenz/TransferLearning/New_Organized_Code/Train_test_results_model/"
#PATH_TO_MODEL = "/glade/u/home/yiwenz/TransferLearning/Train_test_results_model/resnet_11cities.pt"
print(PATH_TO_DATA)
file_to_store='_more_linear_samesize_all_cities_LSTNorm_CL_6_L1_onehot_day_ERA5_US_monthly_0-1_all'
random.seed(42)
train_pct=0.7
number_hdfs_wanted_train=35
number_hdfs_wanted_test=5
EPOCHS = 100
LEARNING_RATE = 1e-3 # 0.0004
DECAY_RATE = 0.96 # 0.9370
DEVICE = "cuda"
BATCH_SIZE = 1024 #128
patience = 5 #10
lam = 0.1 #64.1883

print(torch.cuda.device_count())

#7 channels: ['Red', 'Green', 'Blue', "NIR", "SWIR1","ndbi","ndvi","elevation"]
image_normalize = transforms.Normalize(
                  mean=[0, 0, 0, 0, 0, 0, 0, 2.5839e+02],
                  std=[1, 1, 1, 1, 1, 1, 1, 3.3506e+02]
)

#forcing: ["FLDS", "FSDS", "PRECTmms", "PSRF", "QBOT", "TBOT", "WIND","building height"]
#Added building height in forcing
forcing_mean = torch.from_numpy(np.array([2.7475e+06, 1.1216e+06, 6.3862e-04, 9.7475e+04, 3.8102e+01, 2.9312e+02, 9.8286e-01, 2.6067e-01, 6.6253e+00]))
forcing_std = torch.from_numpy(np.array([6.3123e+05, 2.1583e+05, 3.1177e-03, 5.1533e+03, 1.3052e+01, 1.0468e+01, 2.4544e+00, 2.6655e+00, 1.8508e+00]))

lst_mean = torch.from_numpy(np.array([301.1683]))
lst_std = torch.from_numpy(np.array([10.3701]))

def process_data(forcing, image, month, lst):
    image, forcing, lst= image.to(DEVICE).to(torch.float32), forcing.to(DEVICE), lst.to(DEVICE)
    month = month-1
    month = month.to(DEVICE).to(torch.int64)
    # Image Transformations
#    image[:,7,] = torch.clip(image[:,7,], min=0, max=600)
    image[:,:5,] = torch.clip(image[:,:5,], min=0, max=1)
    image[:,5:7,] = torch.clip(image[:,5:7,], min=-1, max=1)
    image = image_normalize(image)
#    image = image_rotate(image)
    # Forcing Transformation
    forcing = torch.div(torch.sub(forcing, forcing_mean), forcing_std).to(torch.float32)
#    forcing = forcing[:,:7]
#    forcing = forcing.unsqueeze(2).unsqueeze(3)
#    forcing = forcing.repeat(1,1,33,33)
    # LST Transformation
    lst = torch.div(torch.sub(lst, lst_mean), lst_std).to(torch.float32).view(-1, 1)
    lst = lst.view(-1, 1).to(torch.float32)
    return forcing, image, month, lst

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
        forcing_shape = 9,
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
        self.drop1 = nn.Dropout(0.02)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1)

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

    def forward(self, x: Tensor, forcing, one_hot_mon) -> Tensor:
        # See note [TorchScript super()]
#        x = self.conv0(x)
#        x = x.permute([0,3,1,2])
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.avgpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)
        x = self.avgpool2(x)
        x = torch.flatten(x, 1)
#        x = self.drop1(x)
        x = F.leaky_relu(x)
#        x = F.leaky_relu(x)
#        x = self.fc0_1(x)
#        x = F.leaky_relu(x)
#        x = self.fc0_2(x) 
        forcing = self.fc0_1(forcing)
        forcing = F.leaky_relu(forcing)
        forcing = torch.cat((forcing, one_hot_mon), dim=1)
        forcing = self.fc0_2(forcing)
        forcing = F.leaky_relu(forcing)
        forcing = self.fc0_3(forcing)
        x = torch.cat((x, forcing), dim=1)
#        x = self.fc0(x)      
        x = self.fc1(x)
        x = F.leaky_relu(x)        
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)

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

def load_pretrained_weights(model,path):
    new_state_dict = OrderedDict()
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    print('Loaded weights from ',path)
#    print(res.missing_keys)
#    assert set(res.missing_keys) == {"fc.weight", "fc.bias"}, "issue loading pretrained weights"

###############################################################################################################
model=resnet_simplified()
#load_pretrained_weights(model,PATH_TO_MODEL)
pretrain_trainloader, pretrain_testloader, len_train, len_test = BuildPreTrainDataLoaders(
                                                                    path_to_data_dir=PATH_TO_DATA,
                                                                    number_hdfs_wanted_train=number_hdfs_wanted_train,
                                                                    number_hdfs_wanted_test=number_hdfs_wanted_test,  
                                                                     batch_size=BATCH_SIZE, 
                                                                     train_pct=train_pct)

model = model.to(DEVICE)
model = torch.nn.DataParallel(model)
loss_fn_l1 = nn.L1Loss() #loss_fn = RMSELoss
loss_fn = nn.L1Loss()
#loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=patience)

test_loss = []
train_loss = []
test_loss_l1 = []
train_loss_l1 = []
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
    
    if epoch==0:
        tau_running = tau_init
    else:
        tau_running = avg_train_loss
        
    print(f"tau:{tau_running}")

    model.train()
#    loop_train = tqdm(pretrain_trainloader, total=(len_train//BATCH_SIZE) + 1, leave=True)
    for idx, (forcing, image, month, lst) in enumerate(pretrain_trainloader):

        lst_l1=lst.view(-1, 1).to(DEVICE).to(torch.float32)
        forcing, image, month, lst = process_data(forcing, image, month, lst)
        one_hot_mon = F.one_hot(month, num_classes=12)
        optimizer.zero_grad()
        forward_out = model.forward(image, forcing, one_hot_mon)
        loss = loss_fn(forward_out, lst)
        super_loss = SuperLoss(loss,tau_running,lam)
        forward_out_l1=torch.add(torch.multiply(forward_out,lst_std),lst_mean)
        train_l1 = loss_fn_l1(forward_out_l1, lst_l1)
        super_loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        running_train_l1 += train_l1.item()
        train_n_iter += 1
#        loop_train.set_postfix(train_loss=loss.item())


#    loop_test = tqdm(pretrain_testloader, total=(len_test//BATCH_SIZE) + 1, leave=False)

    model.eval()
    with torch.no_grad():
        for idx, (forcing, image, month, lst) in enumerate(pretrain_testloader):
            lst_l1=lst.view(-1, 1).to(DEVICE).to(torch.float32)
            forcing, image, month, lst = process_data(forcing, image, month, lst)
            one_hot_mon = F.one_hot(month, num_classes=12)
            pred = model.forward(image, forcing, one_hot_mon)
            testloss = loss_fn(pred, lst)
            pred_l1=torch.add(torch.multiply(pred,lst_std),lst_mean)
            test_l1 = loss_fn_l1(pred_l1, lst_l1)
            running_test_loss += testloss.item()
            running_test_l1 += test_l1.item()
            test_n_iter += 1
#            loop_test.set_postfix(test_loss=testloss.item())

    avg_train_loss = running_train_loss/train_n_iter
    train_loss.append(avg_train_loss)
    avg_test_loss = running_test_loss/test_n_iter
    test_loss.append(avg_test_loss)
    
    avg_train_l1 = running_train_l1/train_n_iter
    avg_test_l1 = running_test_l1/test_n_iter
    train_loss_l1.append(avg_train_l1)
    test_loss_l1.append(avg_test_l1)
    
    scheduler.step()
    scheduler2.step(avg_test_loss)
    if avg_test_loss < min_test_loss:
        print("Saving Model")
        min_test_loss = avg_test_loss
        torch.save(model.state_dict(), os.path.join(dir_to_store,"best_resnet"+file_to_store+".pt"))
#    torch.save(model.state_dict(), os.path.join(dir_to_store,f"resnet_{file_to_store}_epoch_{epoch}.pt"))

    print("Saving Loss")
    file_name = os.path.join(dir_to_store,"train_loss"+file_to_store+".pkl")
    open_file = open(file_name, "wb")
    pickle.dump(train_loss, open_file)
    open_file.close()
    file_name = os.path.join(dir_to_store,"test_loss"+file_to_store+".pkl")
    open_file = open(file_name, "wb")
    pickle.dump(test_loss, open_file)
    open_file.close()
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
