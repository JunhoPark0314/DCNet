# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
import copy
import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import functional as F


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class _SharedNormBase(nn.Module):
    """
    Common base of _SharedBatchNorm
    Only share weight and bias between shared group
    """
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        num_shared: int = 1,
    ) -> None:
        super(_SharedNormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_shared = num_shared
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros((num_shared, num_features)))
            self.register_buffer("running_var", torch.ones((num_shared, num_features)))
            self.register_buffer("num_batches_tracked", torch.zeros(num_shared, num_features))
        else:
            self.running_mean = [None] * self.num_shared
            self.running_var = [None] * self.num_shared
            self.num_batches_tracked = [None] * self.num_shared

        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            for i in range(self.num_shared):
                # running_mean/running_var/num_batches... are registered at runtime depending
                # if self.track_running_stats is on
                self.running_mean[i].zero_()  # type: ignore[operator]
                self.running_var[i].fill_(1)
                self.num_batches_tracked[i].zero_()  # type: ignore[operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}, num_shared={num_shared}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            for i in range(self.num_shared):
                if num_batches_tracked_key + "_{}".format(i) not in state_dict:
                    state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_SharedNormBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class _SharedBatchNorm(_SharedNormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_shared=1):
        super(_SharedBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, num_shared)

    def forward(self, input: Tensor, share_id: int) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        
        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked[share_id] is not None:  # type: ignore
                self.num_batches_tracked[share_id] = self.num_batches_tracked[share_id] + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked[share_id])
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean[share_id] is None) and (self.running_var[share_id] is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean[share_id] is None or isinstance(self.running_mean[share_id], torch.Tensor)
        assert self.running_var[share_id] is None or isinstance(self.running_var[share_id], torch.Tensor)

        old_mean = copy.deepcopy(self.running_mean)
        old_var = copy.deepcopy(self.running_var)

        result = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean[share_id] if not self.training or self.track_running_stats else None,
            self.running_var[share_id] if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        
        assert self.running_mean[share_id].isfinite().all().item()
        assert self.running_var[share_id].isfinite().all().item()
        return result

class SharedBatchNorm1d(_SharedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class SharedBatchNorm2d(_SharedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SharedBatchNorm3d(_SharedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

