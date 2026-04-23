import cv2
import numpy as np
import torch


# Source: https://github.com/dmMaze/comic-text-detector/blob/440b978563c71b758e31aaa315d100faba1efa2f/utils/imgproc_utils.py#L86C1-L117C31
def letterbox(
    im,
    new_shape=(640, 640),
    color=(0, 0, 0),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=128,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # dw /= 2  # divide padding into 2 sides
    # dh /= 2
    dh, dw = int(dh), int(dw)

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)

    # Source: https://github.com/zyddnys/manga-image-translator/blob/a537cb12b41daf2065795058c2753d87e73fa0fe/manga_translator/detection/ctd.py#L30


def postprocess_mask(img: torch.Tensor | np.ndarray, thresh=None):
    # img = img.permute(1, 2, 0)
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != "cpu":
            img = img.detach().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255

    assert isinstance(img, np.ndarray)
    return img.astype(np.uint8)
