import os

import cv2
import numpy as np

import cvglue

llog = cvglue.setup_logger(name=__name__)


def test_image_parse():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "data", "images", "single_face_img.jpg")
    a = cv2.imread(img_path)
    parser = cvglue.parser.face_parser(model_name="RBF", method="lamply", mode="selfie")
    anno = parser.parse_img(a)
    assert anno["height"] == 513
    assert anno["width"] == 373
    assert anno["channel"] == 3
    assert len(anno["faces"][0]["face_box"]) == 4
    assert len(anno["faces"][0]["key_points"]) == 10
    assert anno["faces"][0]["inout_area"] == 1.0
    assert anno["faces"][0]["confidence"] > 0.99


# Enable only with large local models placed in TORCH_HOME
def test_dataset_parse():
    if not os.environ.get("TORCH_HOME"):
        return

    model1 = os.path.exists(
        os.path.join(os.environ["TORCH_HOME"], "Resnet50_Final.pth")
    )
    model2 = os.path.exists(os.path.join(os.environ["TORCH_HOME"], "WFLW_4HG.pth"))
    model3 = os.path.exists(
        os.path.join(os.environ["TORCH_HOME"], "SDD_FIQA_checkpoints_r50.pth")
    )

    if not model1 or not model2 or not model3:
        llog.warning("Large models are unavailable...Skip test_dataset_parse()")
        return

    def cosine_similarity(A, B):
        A_flat = np.array(A).flatten()
        B_flat = np.array(B).flatten()
        dot_product = np.dot(A_flat, B_flat)
        norm_A = np.linalg.norm(A_flat)
        norm_B = np.linalg.norm(B_flat)
        return dot_product / (norm_A * norm_B)  # 范围 [-1, 1]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "data", "images")
    tmp_json = os.path.join(base_dir, "data", "tmp_annotations.json")
    ref_json = os.path.join(base_dir, "data", "ref_annotations.json")

    parser = cvglue.parser.get_parser("lamply-faceid")
    parser.parse(img_dir, out_json_file=tmp_json)

    anno_dict = cvglue.read_json_file(tmp_json)
    ref_anno_dict = cvglue.read_json_file(ref_json)
    face = anno_dict["single_face_img"]["faces"][0]
    face_ref = ref_anno_dict["single_face_img"]["faces"][0]

    eps = 0.999
    assert cosine_similarity(face["face_box"], face_ref["face_box"]) > eps
    assert cosine_similarity(face["landmarks"], face_ref["landmarks"]) > eps
    assert cosine_similarity(face["key_points"], face_ref["key_points"]) > eps
    assert cosine_similarity(face["headpose"], face_ref["headpose"]) > eps
    assert abs(face["quality"] - face_ref["quality"]) < 1.0
    assert abs(face["blurriness"] - face_ref["blurriness"]) < 10.0
    assert cosine_similarity(face["faceid"], face_ref["faceid"]) > eps

    os.remove(tmp_json)
