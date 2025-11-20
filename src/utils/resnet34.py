import torch
import torch.nn as nn

from typing import Callable, Optional


from torchvision.models import resnet34
from torchvision.models.resnet import ResNet, BasicBlock

from opacus.validators import ModuleValidator


class ReplaceBasicBlock(BasicBlock):
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
        super(ReplaceBasicBlock, self).__init__(
            inplanes, planes, stride, downsample, groups,
            base_width, dilation, norm_layer
        )

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

# opacus hooks are incompatible with inplace operations, change them
class MyResnet34(ResNet):
    def __init__(self):
        super(MyResnet34, self).__init__(ReplaceBasicBlock, [3,4,6,3]) # resnet34 3,4,6,3

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
# opacus hooks are incompatible with inplace operations, change them
def replace_inplace_add(model):
    for name, module in model.named_modules():
        if hasattr(module, 'forward'):
            # For BasicBlock, override the forward method
            if 'BasicBlock' in str(type(module)):
                original_forward = module.forward
                def new_forward(self, x):
                    identity = x
                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)
                    out = self.conv2(out)
                    out = self.bn2(out)
                    if self.downsample is not None:
                        identity = self.downsample(x)
                    # Replace += with regular addition
                    out = out + identity
                    out = self.relu(out)
                    return out
                module.forward = new_forward.__get__(module, module.__class__)
            # e.g., for resnet152 change Bottleneck
            if 'Bottleneck' in str(type(module)): 
                original_forward = module.forward
                def new_forward(self, x):
                        identity = x
                        out = self.conv1(x)
                        out = self.bn1(out)
                        out = self.relu(out)
                        out = self.conv2(out)
                        out = self.bn2(out)
                        out = self.relu(out)
                        out = self.conv3(out)
                        out = self.bn3(out)
                        if self.downsample is not None:
                            identity = self.downsample(x)
                        out = out + identity
                        out = self.relu(out)
                        return out
                module.forward = new_forward.__get__(module, module.__class__)

# opacus hooks are incompatible with inplace operations, change them
def replace_relu_inplace(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.ReLU(inplace=False))
        else:
            replace_relu_inplace(child)  # Recursively apply to submodules
    
# freeze all but the last basic block for private, and non-private training
def resnet34_generator_model(output_dim):  
    # https://colab.research.google.com/drive/18GRMyixn43T_lHF6ZheoAJn4-J8L3LAL#scrollTo=HESocROn1dey
    # https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/6
    if output_dim is None:
        output_dim = 16 
    model = MyResnet34()
    model.load_state_dict(resnet34(weights="IMAGENET1K_V1").state_dict())
    model = ModuleValidator.fix(model) # switch BN to GN
    replace_relu_inplace(model) # switch inplace to False

    resnet_modules = list(model.children())
    backbone = nn.Sequential(*resnet_modules[:-3]) # the first part of resnet model
    head = nn.Sequential(*resnet_modules[-3:-1],
                            nn.Flatten(),
                            nn.Linear(512,output_dim)) # the last part of resnet model (the part we're training)
    backbone = backbone.eval() # don't train this part
    for param in backbone.parameters():
        param.requires_grad = False

    return nn.Sequential(backbone, head)