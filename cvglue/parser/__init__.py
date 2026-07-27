from typing import Union

import numpy as np
import PIL.ImageFile

from .base import base_parser
from .face import (
    attribute_parser,
    attribute_stage,
    blur_parser,
    blur_stage,
    face_parser,
    face_stage,
    faceid_parser,
    faceid_stage,
    genderage_parser,
    genderage_stage,
    landmarks_parser,
    landmarks_stage,
    quality_parser,
    quality_stage,
)

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid "Decompressed Data Too Large" error

__all__ = [
    "anno_exists",
    "attribute_parser",
    "blur_parser",
    "face_parser",
    "faceid_parser",
    "genderage_parser",
    "get_parser",
    "landmarks_parser",
    "quality_parser",
    "set_image_anno",
]


def set_image_anno(base_name, **kwargs):
    return {"name": base_name, **kwargs}


def anno_exists(anno: dict, domain: str | list, **kwargs):
    if isinstance(domain, list):
        logit = [anno_exists(anno, spec) for spec in domain]
        return np.sum(logit) == len(domain)
    if not anno.__contains__(domain):
        return False
    if isinstance(anno[domain], (dict, list)) and len(anno[domain]) == 0:
        return False
    return True


def get_parser(version):
    if "lamply" in version:
        if "faceid" in version:
            stages = [
                face_stage(),
                landmarks_stage(),
                attribute_stage(),
                quality_stage(),
                blur_stage(),
                faceid_stage(),
            ]
            parser = base_parser(stages=stages)
            parser.__version__ = "lamply-2.0-faceid"
            return parser
        elif "mini" in version:
            stages = [
                face_stage(),
                landmarks_stage(),
            ]
            parser = base_parser(stages=stages)
            parser.__version__ = "lamply-2.0-mini"
            return parser
        else:
            stages = [
                face_stage(),
                landmarks_stage(),
                attribute_stage(),
                quality_stage(),
                blur_stage(),
            ]
            parser = base_parser(stages=stages)
            parser.__version__ = "lamply-2.0"
            return parser
    else:
        raise NotImplementedError(version, "is not avaliable.")
