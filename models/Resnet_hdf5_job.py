#make sure about the cuda version
import numpy as np
import webdataset as wds
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms 
import os
import random
from itertools import islice
import pickle
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import h5py
PATH_TO_DATA = "/glade/scratch/yiwenz/transfer_learning_hdf5/phoenix_data.h5" 
#PATH_TO_DATA = "/glade/scratch/yiwenz/TransferLearningData/rand_shard_data/"
dir_to_store="/glade/u/home/yiwenz/TransferLearning/Train_test_results_model/"
file_to_store='_hdf5_test'

#7 channels: ['Red', 'Green', 'Blue', "NIR", "SWIR1","ndbi","ndvi"]
image_normalize = transforms.Normalize(
                  mean=[0, 0, 0, 0, 0, 0, 0],
                  std=[1, 1, 1, 1, 1, 1, 1]
)

#forcing: ["FLDS", "FSDS", "PRECTmms", "PSRF", "QBOT", "TBOT", "WIND"]
forcing_mean = torch.from_numpy(np.array([3.6346e+02, 8.3282e+02, 5.5716e-05, 9.6421e+04, 5.2780e-03, 3.0339e+02, 2.7644e+00]))
forcing_std = torch.from_numpy(np.array([6.5498e+01, 1.7130e+02, 3.7665e-04, 1.0908e+03, 2.9168e-03, 8.6245e+00, 1.5957e+00]))

lst_mean = torch.from_numpy(np.array([315.1010]))
lst_std = torch.from_numpy(np.array([10.9206]))

random.seed(42)
train_pct=0.7
class hdfLoaderTop(Dataset):
    def __init__(self, length=1000000, path=PATH_TO_DATA):
#        path = "/glade/scratch/yiwenz/transfer_learning_hdf5/phoenix_data_test_2.h5"
        self.hf = h5py.File(path, 'r')
        self.key = list(self.hf.keys())[0]
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = self.hf[self.key][index, :]
        forcing = sample[:7]
        month = sample[7]
        lst = sample[8]
        image = sample[9:].reshape(7,33,33)
        return forcing, image, month, lst
    
dataset = hdfLoaderTop()
len_dataset=len(dataset)
train_data, test_data = torch.utils.data.random_split(dataset, [round(len_dataset*train_pct), round(len_dataset*(1-train_pct))])

def process_data(forcing, image, month, lst):
    image, forcing, lst= image.to(DEVICE).to(torch.float32), forcing.to(DEVICE), lst.to(DEVICE)
    month = month-1
    month = month.to(DEVICE).to(torch.int64)
    # Image Transformations
#    image[:,7,] = torch.clip(image[:,7,], min=0, max=600)
    image[:,:5,] = torch.clip(image[:,:5,], min=0, max=1)
    image[:,5:,] = torch.clip(image[:,5:7,], min=-1, max=1)
    image = image_normalize(image)
#    image = image_rotate(image)
    # Forcing Transformation
    forcing = torch.div(torch.sub(forcing, forcing_mean), forcing_std).to(torch.float32)
    forcing = forcing[:,np.array([0,1,2,3,4,6])]
#    forcing = forcing.unsqueeze(2).unsqueeze(3)
#    forcing = forcing.repeat(1,1,33,33)
    # LST Transformation
#     lst = torch.div(torch.sub(lst, lst_mean), lst_std).to(torch.float32).view(-1, 1)
    lst = lst.view(-1, 1).to(torch.float32)
    return forcing, image, month, lst

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
    
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
    
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, r=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
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
        # add SE block
        self.se = SE_Block(planes, r)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # add SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

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
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.leaky_relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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
        in_channel=7,
        forcing_shape = 6,
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
        
#        self.fc0_1 = nn.Linear(128, out_features=256)
#        self.fc0_2 = nn.Linear(256, out_features=1)
        self.fc1 = nn.Linear(128+forcing_shape+12, out_features=256)
        self.drop1 = nn.Dropout(0.02)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

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
        x = torch.cat((x, forcing, one_hot_mon), dim=1)
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


PATH_TO_MODEL="/glade/u/home/yiwenz/TransferLearning/resnet_dropna_qc_cloud_0_1.pt"
def load_pretrained_weights(model,path):
    # if mode == 'online':
    #     """ Loads pretrained weights, and downloads if loading for the first time. """
    #     state_dict = torch.utils.model_zoo.load_url(url)
    #     state_dict.pop("fc.weight")
    #     state_dict.pop("fc.bias")
    #     weight = state_dict['conv1.weight'].clone()
    #     state_dict.pop("conv1.weight")
    #     model.load_state_dict(state_dict, strict=False)
    #     model.conv1.weight.data[:, :3] = weight
    #     model.conv1.weight.data[:, 3] = torch.mean(model.conv1.weight.data[:, :3],dim=1)
    # elif mode == 'local':
    new_state_dict = OrderedDict()
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
#    print(res.missing_keys)
#    assert set(res.missing_keys) == {"fc.weight", "fc.bias"}, "issue loading pretrained weights"

###############################################################################################################
model=resnet_simplified()
#load_pretrained_weights(model,PATH_TO_MODEL)

EPOCHS = 100
LEARNING_RATE = 1e-3
DECAY_RATE = 0.96
DEVICE = "cuda"
BATCH_SIZE = 1024

#167 GB of memory needed
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=48)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=48)

#model=resnet_simplified()

model = model.to(DEVICE)
model = torch.nn.DataParallel(model)
loss_fn = nn.SmoothL1Loss() #loss_fn = RMSELoss
#loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10)

test_loss = []
train_loss = []

lst_mean = lst_mean.to(DEVICE)
lst_std = lst_std.to(DEVICE)
forcing_mean = forcing_mean.to(DEVICE)
forcing_std = forcing_std.to(DEVICE)

min_test_loss = np.inf 
torch.backends.cudnn.benchmark = True
for epoch in range(EPOCHS):
    print("****** EPOCH: [{}/{}] LR: {} ******".format(epoch, EPOCHS, round(optimizer.param_groups[0]['lr'], 6)))
    running_train_loss = 0
    train_n_iter = 0
    running_test_loss = 0
    test_n_iter = 0
    
    # model.train()
    loop_train = tqdm(train_loader, total=(len(train_data)//BATCH_SIZE) + 1, leave=True)
    for idx, (forcing, image, month, lst) in enumerate(loop_train):
        forcing, image, month, lst = process_data(forcing, image, month, lst)
        one_hot_mon = F.one_hot(month, num_classes=12)
        optimizer.zero_grad()
        forward_out = model.forward(image, forcing, one_hot_mon)
        loss = loss_fn(forward_out, lst)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        train_n_iter += 1
#        loop_train.set_postfix(train_loss=loss.item())
        
    loop_test = tqdm(test_loader, total=(len(test_data)//BATCH_SIZE) + 1, leave=False)
    
    # model.eval()
    with torch.no_grad():
        for idx, (forcing, image, month, lst) in enumerate(loop_test):
            forcing, image, month, lst = process_data(forcing, image, month, lst)
            one_hot_mon = F.one_hot(month, num_classes=12)
            pred = model.forward(image, forcing, one_hot_mon)
            testloss = loss_fn(pred, lst)
            running_test_loss += testloss.item()
            test_n_iter += 1
#            loop_test.set_postfix(test_loss=testloss.item())

    avg_train_loss = running_train_loss/train_n_iter
    train_loss.append(avg_train_loss)
    avg_test_loss = running_test_loss/test_n_iter
    test_loss.append(avg_test_loss)
    
    scheduler.step()
    scheduler2.step(avg_test_loss)
    if avg_test_loss < min_test_loss:
        print("Saving Model")
        min_test_loss = avg_test_loss
#        torch.save(model.state_dict(), os.path.join(dir_to_store,"resnet"+file_to_store+".pt"))
    
#    print("Saving Loss")
#    file_name = os.path.join(dir_to_store,"train_loss"+file_to_store+".pkl")
#    open_file = open(file_name, "wb")
#    pickle.dump(train_loss, open_file)
#    open_file.close()
#    file_name = os.path.join(dir_to_store,"test_loss"+file_to_store+".pkl")
#    open_file = open(file_name, "wb")
#    pickle.dump(test_loss, open_file)
#    open_file.close()
    print("------ Train Loss: {}, Test Loss: {} ------".format(avg_train_loss, avg_test_loss))