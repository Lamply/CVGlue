import os

import cv2
from tqdm import tqdm, trange

from ..utils import check_image_format, make_dataset, read_json_file, write_json_file
from . import set_image_anno

__all__ = ["base_parser"]


class base_parser:
    def __init__(self, stages=None):
        self.stages = stages or []
        self.out_json = {}
        self.current_obj_dict = {}
        self.output_dir = "./"
        self.out_json_file = None
        self.append_flag = False
        self.continue_flag = False

    def init_settings(self, out_json_file, append_flag=False, continue_flag=False):
        self.output_dir = os.path.dirname(os.path.abspath(out_json_file))
        self.append_flag = append_flag
        self.continue_flag = continue_flag
        if append_flag or continue_flag:
            self.out_json = read_json_file(out_json_file)
        elif not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"WARNING: output dir ({self.output_dir}) not exists, auto created")
        elif os.path.exists(out_json_file):
            raise RuntimeError(
                f"Unexpected {out_json_file} is already exists, and not work in current mode."
            )
        self.out_json_file = out_json_file

    def saving_json(self):
        write_json_file(self.out_json_file, self.out_json, override=True)

    def parse_img(self, img, name="input_img"):
        try:
            img = check_image_format(img)
        except (cv2.error, ValueError) as e:
            print(name, repr(e))
            return
        self.current_obj_dict = set_image_anno(
            name, height=img.shape[0], width=img.shape[1], channel=img.shape[2]
        )
        for stage in self.stages:
            label_dict = stage(img, self.current_obj_dict)
            if label_dict:
                self.current_obj_dict.update(label_dict)
        self.out_json.update({name: self.current_obj_dict})
        return self.out_json[name]

    def parse_video(self, video_path, out_json_file=None):
        self.out_json_file = (
            out_json_file
            if out_json_file
            else ".".join(video_path.split(".")[:-1]) + ".json"
        )
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_id in trange(frame_count):
            _, frame = capture.read()
            _ = self.parse_img(frame, name=str(frame_id))
        capture.release()
        self.saving_json()

    def parse_file(self, img_path):
        file_name = img_path.split("/")[-1]
        base_name = ".".join(file_name.split(".")[:-1])
        if base_name not in self.out_json or self.append_flag:
            img = cv2.imread(img_path)
            try:
                img = check_image_format(img)
            except (cv2.error, ValueError) as e:
                print(img_path, repr(e))
                return
            self.current_obj_dict = (
                self.out_json[base_name]
                if self.append_flag
                else set_image_anno(
                    base_name,
                    height=img.shape[0],
                    width=img.shape[1],
                    channel=img.shape[2],
                )
            )
            for stage in self.stages:
                label_dict = stage(img, self.current_obj_dict)
                if label_dict:
                    self.current_obj_dict.update(label_dict)
            self.out_json.update({base_name: self.current_obj_dict})
        elif not self.continue_flag:
            raise RuntimeError(
                f"ERROR: label of {file_name} already exists in json file, conflict"
            )

    def parse(
        self,
        img_root_dir,
        out_json_file="./auto_annotations.json",
        append_flag=False,
        continue_flag=False,
    ):
        self.init_settings(
            out_json_file, append_flag=append_flag, continue_flag=continue_flag
        )
        img_paths = sorted(make_dataset(img_root_dir))
        for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
            self.parse_file(img_path)
        self.saving_json()

    # def parse_multi(self, img_root_dir, pid, num_worker=8, out_json_file='./auto_annotations.json', append_flag=False, continue_flag=False):
    #     out_json_file = '.'.join(out_json_file.split('.')[:-1])+'_'+str(pid)+'.json'
    #     self.init_settings(out_json_file, append_flag=append_flag, continue_flag=continue_flag)
    #     img_paths = sorted(make_dataset(img_root_dir))
    #     img_paths = img_paths[pid::num_worker]
    #     for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
    #         self.parse_file(img_path)
    #     self.saving_json()
