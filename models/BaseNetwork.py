import torch.nn as nn
from torch.nn import init



class base_network(nn.Module):

    def __init__(self):
        super(base_network, self).__init__()

        # def init_weights(self, init_type = 'normal', gain=0.02):
            # exit()
