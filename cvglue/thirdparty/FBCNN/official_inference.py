import os.path
import logging
import numpy as np
from datetime import datetime
from collections import OrderedDict
import torch
import cv2
import requests

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
]


def logger_info(logger_name, log_path="default_logger.log"):
    """set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    """
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print("LogHandlers exist!")
    else:
        print("LogHandlers setup!")
        level = logging.INFO
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d : %(message)s", datefmt="%y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), "{:s} is not a valid directory".format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, "{:s} has no valid image file".format(path)
    return images


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths


# convert uint to 4-dimensional torch tensor
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return (
        torch.from_numpy(np.ascontiguousarray(img))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
    )


# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img


# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.0).round())


def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def main():
    testset_name = "Real"  # folder name of real images
    n_channels = 3  # set 1 for grayscale image, set 3 for color image
    model_name = "fbcnn_color.pth"
    nc = [64, 128, 256, 512]
    nb = 4
    testsets = "testsets"
    results = "test_results"

    do_flexible_control = True
    QF_control = [
        5,
        10,
        30,
        50,
        70,
        90,
    ]  # adjust qf as input to provide different results

    result_name = testset_name + "_" + model_name[:-4]
    L_path = os.path.join(testsets, testset_name)
    E_path = os.path.join(results, result_name)  # E_path, for Estimated images
    os.makedirs(E_path, exist_ok=True)

    model_pool = "model_zoo"  # fixed
    model_path = os.path.join(model_pool, model_name)
    if os.path.exists(model_path):
        print(f"loading model from {model_path}")
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{}".format(
            os.path.basename(model_path)
        )
        r = requests.get(url, allow_redirects=True)
        print(f"downloading model {model_path}")
        open(model_path, "wb").write(r.content)

    logger_name = result_name
    logger_info(logger_name, log_path=os.path.join(E_path, logger_name + ".log"))
    logger = logging.getLogger(logger_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    border = 0

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from .model import FBCNN as net

    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode="R")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info("Model path: {:s}".format(model_path))

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnrb"] = []

    L_paths = get_image_paths(L_path)
    for idx, img in enumerate(L_paths):
        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info("{:->4d}--> {:>10s}".format(idx + 1, img_name + ext))
        img_L = imread_uint(img, n_channels=n_channels)

        img_L = uint2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        # img_E,QF = model(img_L, torch.tensor([[0.6]]))
        img_E, QF = model(img_L)
        QF = 1 - QF
        img_E = tensor2single(img_E)
        img_E = single2uint(img_E)
        logger.info("predicted quality factor: {:d}".format(round(float(QF * 100))))
        imsave(img_E, os.path.join(E_path, img_name + ".png"))

        if do_flexible_control:
            for QF_set in QF_control:
                logger.info("Flexible control by QF = {:d}".format(QF_set))
                #    from IPython import embed; embed()
                qf_input = (
                    torch.tensor([[1 - QF_set / 100]]).cuda()
                    if device == torch.device("cuda")
                    else torch.tensor([[1 - QF_set / 100]])
                )
                img_E, QF = model(img_L, qf_input)
                QF = 1 - QF
                img_E = tensor2single(img_E)
                img_E = single2uint(img_E)
                imsave(
                    img_E,
                    os.path.join(E_path, img_name + "_qf_" + str(QF_set) + ".png"),
                )


if __name__ == "__main__":
    main()
