import PIL.ImageFile
from typing import Union
import numpy as np

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid "Decompressed Data Too Large" error

__all__ = [
    "set_image_anno",
    "anno_exists",
    "get_parser",
    "face_parser",
    "landmarks_parser",
    "blur_parser",
    "attribute_parser",
    "genderage_parser",
    "faceid_parser",
    "quality_parser",
]


def set_image_anno(base_name, **kwargs):
    return {"name": base_name, **kwargs}


def anno_exists(anno: dict, domain: Union[str, list], **kwargs):
    if isinstance(domain, list):
        logit = [anno_exists(anno, spec) for spec in domain]
        return np.sum(logit) == len(domain)
    if not anno.__contains__(domain):
        return False
    if isinstance(anno[domain], (dict, list)) and len(anno[domain]) == 0:
        return False
    return True


from .base import base_parser

# from .mask import mask_parser
from .face import (
    face_parser,
    landmarks_parser,
    blur_parser,
    attribute_parser,
    genderage_parser,
    faceid_parser,
    quality_parser,
    face_stage,
    landmarks_stage,
    blur_stage,
    attribute_stage,
    genderage_stage,
    faceid_stage,
    quality_stage,
)


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
