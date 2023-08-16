from torch import nn
import torch

class SpatialMaxPool(nn.Module):

    """ Performs spatial max-pooling on every orientation of the SE2N tensor.
        INPUT:
            - input_tensor in SE2n, a tensor flow tensor with expected shape:
                [BatchSize, nbOrientations, ChannelsIn, Height, Width]
        OUTPUT:
            - output_tensor, the tensor after spatial max-pooling
                [BatchSize, nbOrientations, ChannelsOut, Height/2, Width/2]
    """

    def __init__(self, kernel_size, stride, padding, nbOrientations):
        super().__init__()
        self.nbOrientations = nbOrientations
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, input_tensor):

        # 2D max-pooling is applied to each orientation
        activations = [None] * self.nbOrientations
        for i in range(self.nbOrientations):
            activations[i] = self.pool(input_tensor[:, :, i, :, :])
                # ([28, 8, 8, 61, 61]), nbOrientations = i1
                # [:, :, :, i, :], nbOrientation = i3

        # Re-stack all the pooled activations along the orientation dimension
        tensor_pooled = torch.cat([torch.unsqueeze(t, 2) for t in activations], axis=2)

        return tensor_pooled


class SpatialUpsample(nn.Module):

    """ Performs spatial upsampling on every orientation of the SE2N tensor.
        INPUT:
            - input_tensor in SE2n, a tensor flow tensor with expected shape:
                [BatchSize, nbOrientations, ChannelsIn, Height, Width]
        OUTPUT:
            - output_tensor, the tensor after spatial max-pooling
                [BatchSize, nbOrientations, ChannelsOut, Height/2, Width/2]
    """

    def __init__(self, scale_factor, nbOrientations, mode='nearest', align_corners=None, recompute_scale_factor=None):
        super().__init__()
        self.nbOrientations = nbOrientations
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)

    def forward(self, input_tensor):

        # 2D upsampling is applied to each orientation
        activations = [None] * self.nbOrientations
        for i in range(self.nbOrientations):
            activations[i] = self.upsample(input_tensor[:, :, i, :, :])

        # Re-stack all the upsampled activations along the orientation dimension
        tensor_upsampled = torch.cat([torch.unsqueeze(t, 2) for t in activations], axis=2)

        return tensor_upsampled