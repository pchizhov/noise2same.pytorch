from itertools import product
from typing import Any, List, Optional, Tuple, Union

import albumentations as albu
import numpy as np
from albumentations import Compose

from noise2same.dataset import transforms as t3d

Ints = Optional[Union[int, List[int], Tuple[int, ...]]]


def get_stratified_coords(
    box_size: int, shape: Tuple[int, ...], resample: bool = False,
) -> Tuple[List[int], ...]:
    """
    Create stratified blind spot coordinates
    :param box_size: int, size of stratification box
    :param shape: tuple, image shape
    :param resample: bool, resample if out o box
    :return:
    """
    box_count = [int(np.ceil(s / box_size)) for s in shape]
    coords = []

    for ic in product(*[np.arange(bc) for bc in box_count]):
        sampled = False
        while not sampled:
            coord = tuple(np.random.rand() * box_size for _ in shape)
            coord = [int(i * box_size + c) for i, c in zip(ic, coord)]
            if all(c < s for c, s in zip(coord, shape)):
                coords.append(coord)
                sampled = True
            if not resample:
                break

    coords = tuple(zip(*coords))  # transpose (N, 3) -> (3, N)
    return coords


def mask_like_image(
    image: np.ndarray, mask_percentage: float = 0.5, channels_last: bool = True
) -> np.ndarray:
    """
    Generates a stratified mask of image.shape
    :param image: ndarray, reference image to mask
    :param mask_percentage: float, percentage of pixels to mask, default 0.5%
    :param channels_last: bool, true to process image as channel-last (256, 256, 3)
    :return: ndarray, mask
    """
    # todo understand generator_val
    # https://github.com/divelab/Noise2Same/blob/8cdbfef5c475b9f999dcb1a942649af7026c887b/models.py#L130
    mask = np.zeros_like(image)
    n_channels = image.shape[-1 if channels_last else 0]
    channel_shape = image.shape[:-1] if channels_last else image.shape[1:]
    n_dim = len(channel_shape)
    # I think, here comes a mistake in original implementation (np.sqrt used both for 2D and 3D images)
    # If we use square root for 3D images, we do not reach the required masking percentage
    # See test_dataset.py for checks
    box_size = np.round(np.power(100 / mask_percentage, 1 / n_dim)).astype(np.int)
    for c in range(n_channels):
        mask_coords = get_stratified_coords(box_size=box_size, shape=channel_shape)
        mask_coords = (mask_coords + (c,)) if channels_last else ((c,) + mask_coords)
        mask[mask_coords] = 1.0
    return mask


def training_augmentations_2d(crop: int = 64):
    return Compose(
        [
            albu.RandomCrop(width=crop, height=crop, p=1),
            albu.RandomRotate90(p=0.5),
            albu.Flip(p=0.5),
        ]
    )


def training_augmentations_3d():
    return t3d.Compose(
        [
            t3d.RandomRotate90(p=0.5, axis=(2, 3), channel_axis=(0, 1)),
            t3d.RandomFlip(p=0.5, axis=(2, 3), channel_axis=(0, 1)),
        ]
    )


def _raise(e):
    raise e


class PadAndCropResizer(object):
    """
    https://github.com/divelab/Noise2Same/blob/8cdbfef5c475b9f999dcb1a942649af7026c887b/utils/predict_utils.py#L115
    """

    def __init__(
        self, mode: str = "reflect", div_n: Optional[int] = None, **kwargs: Any
    ):
        self.mode = mode
        self.kwargs = kwargs
        self.pad = None
        self.div_n = div_n

    def _normalize_exclude(self, exclude: Ints, n_dim: int):
        """Return normalized list of excluded axes."""
        if exclude is None:
            return []
        exclude_list = [exclude] if np.isscalar(exclude) else list(exclude)
        exclude_list = [d % n_dim for d in exclude_list]
        len(exclude_list) == len(np.unique(exclude_list)) or _raise(ValueError())
        all((isinstance(d, int) and 0 <= d < n_dim for d in exclude_list)) or _raise(
            ValueError()
        )
        return exclude_list

    def before(self, x: np.ndarray, div_n: int = None, exclude: Ints = None):
        def _split(v):
            a = v // 2
            return a, v - a

        if div_n is None:
            div_n = self.div_n
        assert div_n is not None

        exclude = self._normalize_exclude(exclude, x.ndim)
        self.pad = [
            _split((div_n - s % div_n) % div_n) if (i not in exclude) else (0, 0)
            for i, s in enumerate(x.shape)
        ]
        x_pad = np.pad(x, self.pad, mode=self.mode, **self.kwargs)
        for i in exclude:
            del self.pad[i]
        return x_pad

    def after(self, x: np.ndarray, exclude: Ints = None):

        pads = self.pad[: len(x.shape)]  # ?
        crop = [slice(p[0], -p[1] if p[1] > 0 else None) for p in self.pad]
        for i in self._normalize_exclude(exclude, x.ndim):
            crop.insert(i, slice(None))
        len(crop) == x.ndim or _raise(ValueError())
        return x[tuple(crop)]
