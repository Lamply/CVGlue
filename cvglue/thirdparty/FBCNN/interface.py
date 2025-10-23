import os
import itertools
import torch
from .model import FBCNN
from .comfy_ui_utils import OOM_EXCEPTION, tiled_scale

__all__ = ["FBCNNProcessor"]


class FBCNNProcessor:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model(model_path)

    def _load_model(self, model_path=None):
        n_channels = 3
        if model_path is None:
            model_path = os.path.join(os.environ["TORCH_HOME"], "fbcnn_color.pth")
        elif "gray" in model_path:
            n_channels = 1

        fbcnn = FBCNN(in_out_channels=n_channels)
        fbcnn.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        fbcnn.eval().requires_grad_(False)
        fbcnn.to(self.device)
        return fbcnn

    # def _tiled_scale(self, image, process_func, tile_x=1024, tile_y=1024, overlap=32,
    #                upscale_amount=1, pbar=None):
    #     """分块处理图像的简化实现"""
    #     _, _, h, w = image.shape
    #     output = torch.zeros_like(image)

    #     for y in range(0, h, tile_y - overlap):
    #         for x in range(0, w, tile_x - overlap):
    #             # 提取图块
    #             tile = image[:, :, y:y+tile_y, x:x+tile_x]

    #             # 处理图块
    #             processed_tile = process_func(tile)

    #             # 将处理后的图块放回输出图像
    #             output[:, :, y:y+tile_y, x:x+tile_x] = processed_tile

    #             # 更新进度条
    #             if pbar:
    #                 pbar.update(1)

    #     return output

    def process_image(
        self, image, auto_detect=True, compression_level=0, tile_size=1024, overlap=32
    ):
        """
        处理图像

        Args:
            image: 输入图像（PyTorch张量，形状为[C,H,W]或[B,C,H,W]），值域为 [0.0, 1.0]
            auto_detect: 是否自动检测压缩质量
            compression_level: 压缩级别（0-100）
            tile_size: 分块大小
            overlap: 块重叠大小

        Returns:
            处理后的图像（PyTorch张量）
        """
        # 确保输入格式正确
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # 移动设备
        image = image.to(self.device)

        # 设置质量因子
        qf = None
        if not auto_detect:
            qf = torch.tensor(
                [[1 - compression_level / 100]], dtype=torch.float32, device=self.device
            )

        # def process_tile(tile):
        #     return self.model(tile, qf)

        # # 使用简化版的分块处理
        # _, _, h, w = image.shape
        # steps = ((h - 1) // (tile_size - overlap) + 1) * ((w - 1) // (tile_size - overlap) + 1)

        # result = self._tiled_scale(
        #     image, process_tile,
        #     tile_x=tile_size, tile_y=tile_size,
        #     overlap=overlap
        # )

        tile = 1024
        overlap = 32

        oom = True
        while oom:
            try:
                result = tiled_scale(
                    image,
                    lambda a: self.model.forward(a, qf),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=1,
                )
                oom = False
            except OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        # 限制输出范围
        result = torch.clamp(result, min=0, max=1.0)

        return result.squeeze(0) if result.shape[0] == 1 else result
