import os
import glob
import re
import unicodedata
import json
import requests
import numpy as np
from typing import Union, List, Set, Iterable

try:
    from deepdiff import DeepDiff

    # pip install u-msgpack-python
    import umsgpack
except:
    pass

from .logger import setup_logger

llog = setup_logger(name=__name__)

SUPPORTED_IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".bmp",
    ".BMP",
    ".webp",
    ".WEBP",
]

SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".MP4", ".flv", ".FLV"]

__all__ = [
    "check_aligned_img_dataset",
    "check_dict_differences",
    "check_grouped_img_dataset",
    "check_if_overlap_files",
    "check_image_format",
    "convert_safe_filename",
    "download_url",
    "get_base_name",
    "get_ext_name",
    "get_file_name",
    "make_dataset",
    "make_grouped_dataset",
    "make_structural_dataset",
    "read_json_file",
    "select_subdataset",
    "select_subdataset_idxs",
    "write_json_file",
    "make_dataset_recursive",
    "SUPPORTED_IMG_EXTENSIONS",
]


def get_file_name(file_path):
    file_name = os.path.basename(file_path)
    return file_name


def get_base_name(file_path):
    file_name = get_file_name(file_path)
    split_name = file_name.split(".")
    base_name = ".".join(split_name[:-1]) if len(split_name) > 1 else split_name[0]
    return base_name


def get_ext_name(file_path):
    try:
        ext_name = os.path.splitext(file_path)[-1]
    except:
        ext_name = None
    return ext_name


def convert_safe_filename(file_name):
    # 只保留字母、数字、点、下划线和连字符
    # \u4e00-\u9fa5 用于保留中文字符（如果需要）
    return re.sub(r"[^a-zA-Z0-9\._\-\u4e00-\u9fa5]", "-", file_name)


def slugify(file_name: str, lower: bool = True, ascii: bool = True) -> str:
    """
    将字符串转换为安全的 Linux/Web 文件名。
    1. 规范化 Unicode
    2. 转换为小写（默认，可选）
    3. 替换非字母、数字、下划线、连字符为下划线
    4. 将空格、连字符转换为下划线
    """
    file_name = str(file_name)
    # 将 Unicode 字符拆解为基础字符（例如 将 'é' 转为 'e'）
    file_name = unicodedata.normalize("NFKD", file_name)
    # （可选）移除重音符号等非 ASCII 字符
    if ascii:
        file_name = file_name.encode("ascii", "ignore").decode("ascii")
    # 替换非字母、数字、下划线、连字符
    file_name = re.sub(r"[^\w\s-]", "_", file_name).strip()
    # （可选）转换为小写
    if lower:
        file_name = file_name.lower()
    # 将空格和重复的连字符替换为单个下划线
    return re.sub(r"[-\s]+", "_", file_name)


def make_dataset(dir):
    paths = []
    for ext in SUPPORTED_IMG_EXTENSIONS:
        paths += glob.glob(os.path.join(dir, "*" + ext))
    return paths


def select_subdataset(dataset, subset_list, path2name=False):
    name_list = [get_base_name(p) for p in subset_list] if path2name else subset_list
    return [p for p in dataset if get_base_name(p) in name_list]


def select_subdataset_idxs(dataset, subset_list, path2name=False):
    name_list = [get_base_name(p) for p in subset_list] if path2name else subset_list
    return [i for i, p in enumerate(dataset) if get_base_name(p) in name_list]


def make_grouped_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    sequences_dir = sorted(os.listdir(dir))
    for seq in sequences_dir:
        seq_dir = os.path.join(dir, seq)
        if os.path.isdir(seq_dir) and seq[0] != ".":
            paths = sorted(make_dataset(seq_dir))
            if len(paths) > 0:
                images.append(paths)
    return images


def make_structural_dataset(root_dir, leaf_depth):
    """Build structural dataset which images place in the leaf of directory tree.

    Args:
        root_dir(str):            root directory
        leaf_depth(int):          depth of leaf

    Outs:
        dataset(dict)
    """

    def recursive_list(cur_dir, cur_depth=0):
        cur_depth += 1
        if cur_depth == leaf_depth:
            return sorted(make_dataset(cur_dir))
        dataset_tree = {}
        for file_name in os.listdir(cur_dir):
            file_path = os.path.join(cur_dir, file_name)
            if os.path.isdir(file_path) and file_name[0] != ".":
                dataset_tree[file_name] = recursive_list(
                    os.path.join(cur_dir, file_name), cur_depth
                )
        return dataset_tree

    dataset = recursive_list(root_dir)
    return dataset


def make_dataset_recursive(
    root_path: str,
    extensions: Union[str, Iterable[str]],
    exclude_dirs: Union[Iterable[str], None] = None,
    exclude_files: Union[Iterable[str], None] = None,
) -> List[str]:
    """
    递归获取指定目录下所有包含指定后缀的文件的绝对路径，支持排除特定目录和文件。

    :param root_path: 要搜索的根目录路径
    :param extensions: 需要查找的文件后缀，可以是字符串 (如 '.txt') 或列表/元组 (如 ['.py', '.json'])
    :param exclude_dirs: 需要排除的目录名称列表 (完全匹配，如 ['node_modules', '.git'])
    :param exclude_files: 需要排除的文件名称列表 (完全匹配，如 ['config.py', 'temp.txt'])
    :return: 包含绝对路径的列表
    """
    # 1. 参数格式化与预处理
    # 确保 extensions 是一个元组，并且以 '.' 开头 (str.endswith 需要元组)
    if isinstance(extensions, str):
        extensions = (extensions,)
    extensions = tuple(ext if ext.startswith(".") else f".{ext}" for ext in extensions)

    exclude_dirs_set: Set[str] = set(exclude_dirs) if exclude_dirs else set()
    exclude_files_set: Set[str] = set(exclude_files) if exclude_files else set()

    root_abs_path = os.path.abspath(root_path)
    result_paths: List[str] = []

    # 2. 遍历目录树
    for dirpath, dirnames, filenames in os.walk(root_abs_path):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs_set]

        # 3. 筛选文件
        for filename in filenames:
            # 排除特定文件
            if filename in exclude_files_set:
                continue

            # 匹配后缀
            if filename.endswith(extensions):
                # 拼接并保存绝对路径
                abs_path = os.path.join(dirpath, filename)
                result_paths.append(abs_path)

    return result_paths


def check_if_overlap_files(paths):
    from collections import Counter

    base_names = [get_base_name(path) for path in paths]
    count_list = dict(Counter(base_names))
    return {key: value for key, value in count_list.items() if value > 1}


def check_image_format(img, allow_float=False, fix_channels=True):
    if img is None:
        raise RuntimeError("Image is empty!")

    if fix_channels:
        if len(img.shape) == 2 or len(img.shape) == 3:
            img = img.reshape([img.shape[0], img.shape[1], -1])
        else:
            raise ValueError(f"Image is not regular image! img.shape: {img.shape}")

    if img.dtype == np.uint8:
        pass
    elif img.dtype in (np.float32, np.float64):
        if not allow_float:
            llog.info(
                f"Image data type is not uint8 and {img.dtype} is not allowed, convert to uint8 format"
            )
            img = np.uint8(np.clip(img, 0, 255))
    else:
        raise TypeError(f"Image data type {img.dtype} is unexpected.")

    return img


def check_aligned_img_dataset(A_paths, B_paths):
    if len(A_paths) == 0:
        raise Exception("Dataset A is empty, please check the `dataroot` option.")
    if len(A_paths) != len(B_paths):
        raise ValueError(
            "Different size of A=%d and B=%d " % (len(A_paths), len(B_paths))
        )
    for i in range(len(A_paths)):
        A_name = get_base_name(A_paths[i])
        B_name = get_base_name(B_paths[i])
        if A_name != B_name:
            raise ValueError(
                "A and B names is not aligned: {}, {}".format(A_paths[i], B_paths[i])
            )


def check_grouped_img_dataset(A_paths, B_paths):
    if len(A_paths) == 0:
        raise Exception("Dataset is empty, please check the `dataroot` option.")
    if len(A_paths) != len(B_paths):
        raise ValueError(
            "Different size of A=%d and B=%d " % (len(A_paths), len(B_paths))
        )
    for i in range(len(A_paths)):
        check_aligned_img_dataset(A_paths[i], B_paths[i])


def download_url(url, out_path):
    with open(out_path, "wb") as f:
        r = requests.get(url, timeout=None, verify=False)
        f.write(r.content)


def read_json_file(file, use_msgpack=False, **open_kwargs):
    with open(file, "rb" if use_msgpack else "r", **open_kwargs) as f:
        out = umsgpack.unpack(f) if use_msgpack else json.load(f)
    return out


def write_json_file(
    file,
    obj,
    override=False,
    use_msgpack=False,
    min_size=10,
    encoding="utf-8",
    **dump_kwargs,
):
    """Serializes an object and writes it to a file using JSON or MessagePack.

    Args:
        file (str): Path to the target file.
        obj (Any): The Python object to serialize.
        override (bool): If False, prevents overwriting an existing file. Defaults to False.
        use_msgpack (bool): If True, serializes to MessagePack binary format instead of JSON text. Defaults to False.
        min_size (int): Minimum expected length (chars/bytes) of serialized data to guard against empty dumps. Defaults to 10.
        encoding (str): Text encoding used strictly for JSON files. Defaults to 'utf-8'.
        **dump_kwargs: Optional keyword arguments passed directly to `json.dumps()`.
            - For better visualize: "indent=2, ensure_ascii=False"

    Raises:
        RuntimeError: If the file exists and `override` is False.
        RuntimeError: If the serialized content size is less than or equal to `min_size`.
        ValueError: If an encoding is mistakenly passed into binary file mode.
    """
    if os.path.exists(file) and not override:
        raise RuntimeError(f"Try to override file with override={override}.")

    dump_cont = umsgpack.packb(obj) if use_msgpack else json.dumps(obj, **dump_kwargs)

    if len(dump_cont) <= min_size:
        raise RuntimeError(
            f"Dump object size is smaller than {min_size}, which is not expected."
        )

    open_kwargs = {"mode": "wb+" if use_msgpack else "w+"}
    if not use_msgpack:
        open_kwargs["encoding"] = encoding

    with open(file, **open_kwargs) as f:
        f.write(dump_cont)


def check_dict_differences(
    src_dict: dict,
    ref_dict: dict,
    exclude_diff: List[str] = None,
    exclude_keys: List[str] = None,
) -> None:
    """
    Check the differences between two dictionaries.

    Args:
        src_dict(dict):          Source dictionary to compare
        ref_dict(dict):          Reference dictionary to compare
        exclude_diff(List[str]): List of differences types to ignore in the checking process
                                 (defaults to None)
        exclude_keys(List[str]): List of keys to exclude from comparison (defaults to None)

    Raises:
        ValueError: If there are differences between the dictionaries that are not excluded
    """
    if exclude_diff is None:
        exclude_diff = []
    if exclude_keys is None:
        exclude_keys = []

    dict_diff = DeepDiff(src_dict, ref_dict)
    # print(dict_diff.pretty())

    def log_errors(diff_type, diff_data):
        if diff_type in ["dictionary_item_removed", "dictionary_item_added"]:
            return [
                entity
                for entity in diff_data
                if all(ex_key not in entity for ex_key in exclude_keys)
            ]
        else:
            return [
                f"{name} {entity}"
                for name, entity in diff_data.items()
                if all(ex_key not in name for ex_key in exclude_keys)
            ]

    for diff_type, diff_data in dict_diff.items():
        if diff_type in exclude_diff:
            continue
        errors = log_errors(diff_type, diff_data)
        if len(errors) > 0:
            raise ValueError(f"Unexpected {diff_type}: {errors}.")
