from __future__ import annotations

from .base import anno_exists, base_parser, set_image_anno
from .face import (
    attribute_stage,
    blur_stage,
    face_stage,
    faceid_stage,
    genderage_stage,
    landmarks_stage,
    quality_stage,
)

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

def face_parser(method="lamply", mode="selfie", **kwargs):
    return base_parser(stages=[face_stage(method=method, mode=mode, **kwargs)])

def landmarks_parser(method="adaptivewing"):
    return base_parser(stages=[landmarks_stage(method=method)])

def attribute_parser(method="headpose"):
    return base_parser(stages=[attribute_stage(method=method)])

def genderage_parser(method="insightface"):
    return base_parser(stages=[genderage_stage(method=method)])

def faceid_parser(method="insightface"):
    return base_parser(stages=[faceid_stage(method=method)])

def blur_parser(method="opencv"):
    return base_parser(stages=[blur_stage(method=method)])

def quality_parser(method="tface"):
    return base_parser(stages=[quality_stage(method=method)])

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