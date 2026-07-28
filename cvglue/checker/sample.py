import json
from collections import defaultdict

from .base import base_checker

__all__ = [
    "remove_sample_checker",
    "sample_checker",
    "sample_once_checker",
    "single_uid_checker",
]


class single_uid_checker(base_checker):
    """Multi-processing is not supported"""

    def __init__(self, uid_idx=0, exclude=None, verbose=False):
        super().__init__()
        self.uid_idx = uid_idx
        self.exclude = exclude
        self.uid_dict = defaultdict(list)
        self.verbose = verbose

    def check_data(self, iap_data):
        uid = iap_data[1]["name"].split("_")[self.uid_idx]
        if uid not in self.uid_dict or uid == self.exclude:
            self.uid_dict[uid].append(iap_data[1]["name"])
            return True
        self.uid_dict[uid].append(iap_data[1]["name"])
        return False


class sample_checker(base_checker):
    """Multi-processing is not supported"""

    def __init__(self, name_list, sample_inlier=True, verbose=False):
        super().__init__()
        self.sample = sample_inlier
        self.verbose = verbose
        if isinstance(name_list, list):
            self.uid_list = name_list
        elif isinstance(name_list, str):
            with open(name_list) as f:
                self.uid_list = json.load(f)
        else:
            raise TypeError("name_list must be a list of strings or a JSON file path; got " + repr(type(name_list).__name__))

    def check_data(self, iap_data):
        image_name = iap_data[1]["name"]
        if image_name in self.uid_list:
            if self.verbose:
                print(f"removing sample {image_name}")
            return self.sample
        return not self.sample


# backward compatible
class remove_sample_checker(sample_checker):
    def __init__(self, outlier_list, verbose=False):
        super().__init__(outlier_list, sample_inlier=False, verbose=verbose)


class sample_once_checker(base_checker):
    """Sample key-value once in a dictionary of list

    Notice:
        - Multi-processing is not supported

    Args:
        list_dict(dict):            format like { key1:[x1,x2,x3], key2:[x4,x5,x6] }
    """

    def __init__(self, list_dict, exclude=None, verbose=False):
        super().__init__()
        if isinstance(list_dict, dict):
            self.list_dict = list_dict
        elif isinstance(list_dict, str):
            with open(list_dict) as f:
                self.list_dict = json.load(f)
        else:
            raise TypeError(
                f"list_dict must be a dict or a JSON file path; got {type(list_dict)}"
            )
        self.appeared = []
        self.exclude = exclude
        self.verbose = verbose

    def check_data(self, iap_data):
        base_name = iap_data[1]["name"]
        for key in self.list_dict:
            if base_name in self.list_dict[key]:
                break
        if key in self.appeared:
            return False
        self.appeared.append(key)
        return True
