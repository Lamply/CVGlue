import os

from ..utils import setup_logger
from .base import base_parser

__all__ = [
    "attribute_parser",
    "attribute_stage",
    "blur_parser",
    "blur_stage",
    "face_parser",
    "face_stage",
    "faceid_parser",
    "faceid_stage",
    "genderage_parser",
    "genderage_stage",
    "landmarks_parser",
    "landmarks_stage",
    "quality_parser",
    "quality_stage",
]

llog = setup_logger(name=__name__)


class face_stage:
    def __init__(self, method="lamply", mode="selfie", **kwargs):
        if method == "lamply":
            self.set_lamply_detector(mode, **kwargs)
        elif method == "insightface":
            self.set_insightface_detector()
        else:
            raise NotImplementedError(
                "method %s is not implemented, choose one of [lamply, insightface]"
                % method
            )

    def set_lamply_detector(self, mode, **kwargs):
        from ..detector import face_detector

        self.fd = face_detector(detect_mode=mode, **kwargs)
        self.detect_func = self.face_bbox_keypoints_detector

    def set_insightface_detector(self):
        from insightface.app import FaceAnalysis  # type: ignore

        self.app = FaceAnalysis(
            root=os.environ["TORCH_HOME"],
            providers=["CUDAExecutionProvider"],
        )
        self.app.prepare(ctx_id=0)
        self.detect_func = self.face_analysis_detector

    def __call__(self, img, context_dict):
        return self.detect_func(img, context_dict)

    def face_bbox_keypoints_detector(self, img, context_dict):
        face_list = []
        try:
            dets = self.fd(img)
            face_cnt = dets.shape[0] if len(dets) > 0 else 0
        except Exception as e:
            llog.warning(f"face_bbox_keypoints_detector failed with: {e}")
            return {}

        def cal_face_area(face_box):
            return (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])

        for i in range(face_cnt):
            face_box = list(dets[i, :4])
            lx = max(0, face_box[0])
            ly = max(0, face_box[1])
            rx = min(img.shape[1], face_box[2])
            ry = min(img.shape[0], face_box[3])
            area_after_fix = cal_face_area((lx, ly, rx, ry))
            area_before_fix = cal_face_area(face_box)
            if area_before_fix > 0:
                inout_area = area_after_fix / area_before_fix
            else:
                inout_area = 0.0
            face_list += [
                {
                    "face_box": face_box,
                    "inout_area": inout_area,
                    "confidence": dets[i, 4],
                    "key_points": list(dets[i, 5:]),
                }
            ]

        return {"faces": face_list}

    def face_analysis_detector(self, img, context_dict):
        try:
            face_list = self.app.get(img)
        except Exception as e:
            llog.warning(repr(e))
            return {}
        return {"faces": face_list}


def face_parser(method="lamply", mode="selfie", **kwargs):
    return base_parser(stages=[face_stage(method=method, mode=mode, **kwargs)])


class landmarks_stage:
    def __init__(self, method="adaptivewing"):
        if method == "adaptivewing":
            self.set_adaptivewing_detector()
        else:
            raise NotImplementedError(
                "method %s is not implemented, choose one of [adaptivewing]" % method
            )

    def set_adaptivewing_detector(self):
        from ..detector import landmark_detector

        self.ld = landmark_detector()
        self.detect_func = self.adaptivewing_detector

    def __call__(self, img, context_dict):
        return self.detect_func(img, context_dict)

    def adaptivewing_detector(self, img, context_dict):
        base_name = ""
        try:
            base_name = context_dict.get("name", "")
            update_dict = context_dict.get("faces", [])
            for face in update_dict:
                face_box = face["face_box"]
                lands = self.ld(img, face_box)
                face["landmarks"] = lands.tolist()
        except Exception as e:
            llog.warning(f"WARNING: {base_name} landmarks stage failed: {e}")
        return {}


def landmarks_parser(method="adaptivewing"):
    return base_parser(stages=[landmarks_stage(method=method)])


class attribute_stage:
    def __init__(self, method="headpose"):
        if method == "headpose":
            self.set_headpose_detector()
        else:
            raise NotImplementedError(
                "method %s is not implemented, choose one of [headpose]" % method
            )

    def set_headpose_detector(self):
        from ..detector import attribute_detector

        self.ad = attribute_detector()
        self.detect_func = self.face_attribute_detector

    def __call__(self, img, context_dict):
        return self.detect_func(img, context_dict)

    def face_attribute_detector(self, img, context_dict):
        base_name = ""
        try:
            base_name = context_dict.get("name", "")
            update_dict = context_dict.get("faces", [])
            for face in update_dict:
                headpose_pyr = self.ad(img, face["face_box"])
                face["headpose"] = headpose_pyr
        except Exception as e:
            llog.warning(f"WARNING: {base_name} attribute stage failed: {e}")
        return {}


def attribute_parser(method="headpose"):
    return base_parser(stages=[attribute_stage(method=method)])


class genderage_stage:
    def __init__(self, method="insightface"):
        if method == "insightface":
            self.set_genderage_detector()
        else:
            raise NotImplementedError(
                "method %s is not implemented, choose one of [headpose]" % method
            )

    def set_genderage_detector(self):
        import insightface  # type: ignore
        from insightface.app import FaceAnalysis  # type: ignore

        self.app = FaceAnalysis(
            root=os.environ["TORCH_HOME"], providers=["CUDAExecutionProvider"]
        )
        self.app.prepare(ctx_id=0)
        self.face_dict = insightface.app.common.Face({"bbox": None})
        self.detect_func = self.face_genderage_detector

    def __call__(self, img, context_dict):
        return self.detect_func(img, context_dict)

    def face_genderage_detector(self, img, context_dict):
        base_name = ""
        try:
            base_name = context_dict.get("name", "")
            update_dict = context_dict.get("faces", [])
            for face in update_dict:
                self.face_dict["bbox"] = face["face_box"]
                gender, age = self.app.models["genderage"].get(img, self.face_dict)
                face["gender"] = int(gender)
                face["age"] = int(age)
        except Exception as e:
            llog.warning(f"WARNING: {base_name} genderage stage failed: {e}")
        return {}


def genderage_parser(method="insightface"):
    return base_parser(stages=[genderage_stage(method=method)])


class faceid_stage:
    def __init__(self, method="insightface"):
        if method == "insightface":
            self.set_faceid_detector()
        else:
            raise NotImplementedError(
                "method %s is not implemented, choose one of [headpose]" % method
            )

    def set_faceid_detector(self):
        from ..detector import faceid_detector

        self.faid_detector = faceid_detector("model_ir_se50")
        self.detect_func = self.face_id_detector

    def __call__(self, img, context_dict):
        return self.detect_func(img, context_dict)

    def face_id_detector(self, img, context_dict):
        base_name = ""
        try:
            base_name = context_dict.get("name", "")
            update_dict = context_dict.get("faces", [])
            for face in update_dict:
                face["faceid"] = (
                    self.faid_detector(img, face["key_points"]).flatten().tolist()
                )
        except Exception as e:
            llog.warning(f"WARNING: {base_name} faceid stage failed: {e}")
        return {}


def faceid_parser(method="insightface"):
    return base_parser(stages=[faceid_stage(method=method)])


class blur_stage:
    def __init__(self, method="opencv"):
        if method == "opencv":
            self.set_blur_detector()
        else:
            raise NotImplementedError(
                "method %s is not implemented, choose one of [insightface]" % method
            )

    def set_blur_detector(self):
        from ..checker import blur_checker

        self.blur_detector = blur_checker()
        self.detect_func = self.blur_detect

    def __call__(self, img, context_dict):
        return self.detect_func(img, context_dict)

    def blur_detect(self, img, context_dict):
        try:
            base_name = context_dict.get("name", "")
            update_dict = context_dict.get("faces", [])
            for face in update_dict:
                self.blur_detector.check_image(img, face)
                face["blurriness"] = self.blur_detector.score
        except Exception as e:
            llog.warning(f"WARNING: {base_name} blur stage failed: {e}")
        return {}


def blur_parser(method="opencv"):
    return base_parser(stages=[blur_stage(method=method)])


class quality_stage:
    def __init__(self, method="tface"):
        if method == "tface":
            self.set_face_quality_detector()
        else:
            raise NotImplementedError(
                "method %s is not implemented, choose one of [headpose]" % method
            )

    def set_face_quality_detector(self):
        from ..detector import quality_detector

        self.qd = quality_detector("r50")
        self.detect_func = self.face_quality_detector

    def __call__(self, img, context_dict):
        return self.detect_func(img, context_dict)

    def face_quality_detector(self, img, context_dict):
        base_name = ""
        try:
            base_name = context_dict.get("name", "")
            update_dict = context_dict.get("faces", [])
            for face in update_dict:
                face["quality"] = self.qd(img, face["key_points"]).item()
        except Exception as e:
            llog.warning(f"WARNING: {base_name} quality stage failed: {e}")
        return {}


def quality_parser(method="tface"):
    return base_parser(stages=[quality_stage(method=method)])
