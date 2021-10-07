# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import numbers
import torch
from torch import nn
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        if target is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is None:
                return image
            target = target.transpose(0)
        if target is None:
            return image
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target=None):
        image = self.color_jitter(image)
        if target is None:
            return image
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.crop = torchvision.transforms.transforms.RandomCrop(size)
    def __call__(self, image, target=None):
        image = self.crop(image)
        if target is None:
            return image
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        if target is None:
            return F.to_tensor(image)
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

class ResizeWiMin(nn.Module):
    def __init__(self, min_size, max_size):
        super(ResizeWiMin, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def forward(self, image):
        if min(list(image.shape[-2:])) < self.min_size:
            size = self.get_size(image.shape[-2:])
            image = F.resize(image, size)
        return image


class NRandomCrop(nn.Module):

    def __init__(self, size, n=1, padding=0, pad_if_needed=False):
        super(NRandomCrop, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.n = n

    @staticmethod
    def get_params(img, output_size, n):
        h, w = img.shape[2:]
        th, tw = output_size
        if w == tw and h == th:
            return [0] * n, [0] * n, h, w

        try:
            i_list = [random.randint(0, h - th) for i in range(n)]
            j_list = [random.randint(0, w - tw) for i in range(n)]
        except:
            print(h, th, w, tw)
            exit()

        return i_list, j_list, th, tw

    def forward(self, img):
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size, self.n)

        return n_random_crops(img, i, j, h, w)


    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def n_random_crops(img, y, x, h, w):

    crops = []
    for i in range(len(x)):
        new_crop = img[...,y[i]:y[i]+h, x[i]:x[i]+w]
        #new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
        crops.append(new_crop)
    return torch.cat(crops)