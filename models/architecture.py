import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResnetGenerator(nn.Module):

    def __init__(self, input_ch, output_ch, num_filters=64, norm_layer=nn.InstanceNorm2d, num_blocks=9, padding_type='reflect'):
        """
        Parameters:
            input_ch    : number of channels in input images
            output_ch   : number of channels in output images
            num_filters : number of filters in the last convolution layer
            norm_layer  : type of normalization layer
            num_blocks  : the number of ResnetBlocks
        """

        assert(num_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.num_filters = num_filters

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2(input_ch, num_filters, kernel_size=7, padding=0, bias=True),
                 norm_layer(num_filters),
                 nn.ReLU(True)]

        num_downsampling = 2


        # add downsampling layers
        for idx in range(num_downsampling):
            mult = 2 ** idx
            model += [nn.Conv2d(num_filters * mult, num_filters * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                      norm_layer(num_filters * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** num_downsampling

        for idx in range(num_blocks):
            model += [ResnetBlock(num_filters * mult, padding_type=padding_type, norm_layer=norm_layer)]

        for idx in range(num_downsampling):
            mult = 2 ** (num_downsampling - idx)
            model += [nn.ConvTranspose2d(num_filters * mult, int(num_filters * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      norm_layer(int(num_filters * mult / 2)),
                      nn.ReLU(True)]

        model += nn.ReflectionPad2d(3)
        model += nn.Conv2d(num_filters, output_ch, kernel_size=7, padding=0)
        model += [nn.Tanh()]

        self.resnet_generator = nn.Sequential(*model)

    def forward(self, x):
        return self.resnet_generator(x)


class ResnetBlock(nn.Module):

     def __init__(self, dim, padding_type, norm_layer, use_dropout=True, use_bias=True):
         super(ResnetBlock, self).__init__()

         block = []

         p = 0
         if padding_type == 'reflect':
             block += [nn.ReflectionPad2d(1)]

         block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim),
                   nn.ReLU(True)]

         if use_dropout:
             block += [nn.Dropout(0.5)]

         p = 0
         if padding_type == 'reflect':
             block += [nn.ReflectionPad2d(1)]

         block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim),
                   nn.ReLU(True)]

         self.resnetblock = nn.Sequential(*block)

     def forward(self, x):
         y = self.resnetblock(x)
         out = x + y

         return out


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
