import numpy as np
import SimpleITK as sitk

from frida import Pipeline
from frida.readers import ReadVolume
from frida.cast import ToNumpyArray
from frida.augmentations import Augmentation, RandomLinearDisplacement
from frida.transforms import PadAndCropTo, ZeroOneScaling

import torchvision.datasets as datasets

from logging import getLogger

logger = getLogger()


class HistogramDistortion(Augmentation):

    def __init__(self, probability=0.5):
        self.p = probability
        super(HistogramDistortion, self).__init__()

    def __call__(self, image):
        np.random.seed(self.random_seed)
        x = sitk.GetArrayFromImage(image)
        if np.random.uniform() > self.p:
            a = np.random.uniform()
            x = x*a + np.abs(x)*(1-a)
        if np.random.uniform() > self.p:
            x = np.log(np.abs(x)+1.)
        image = sitk.GetImageFromArray(x)
        return image


class RadiologyDataset(datasets.DatasetFolder):

    def __init__(
            self,
            data_path,
            loader,
            size_crops,
            nmb_crops,
            size_dataset=-1,
            return_index=False,
    ):
        super(RadiologyDataset, self).__init__(
            data_path, loader=loader, extensions='nrrd')

        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.return_index = return_index
        self.random_seed = 0

        self.loader = Pipeline(
            ReadVolume()
        )

        displacement = RandomLinearDisplacement(
            rotation_range=10, shift_range=.1, zoom_range=.1)
        distortion = HistogramDistortion()

        trans = Pipeline(
            displacement,
            PadAndCropTo((128, 128, 128)),
            distortion,
            ZeroOneScaling(),
            ToNumpyArray(add_batch_dim=False, add_singleton_dim=True)
        )

        self.trans = trans

    def __make_crop__(self, image):
        self.trans.steps[0].random_seed = self.random_seed
        self.trans.steps[2].random_seed = self.random_seed
        self.random_seed += 1
        return self.trans(image)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = [self.__make_crop__(image) for _ in range(self.nmb_crops)]
        if self.return_index:
            return index, multi_crops
        return multi_crops