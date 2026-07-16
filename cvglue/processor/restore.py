import os
from ..utils.datautils import to_tensor, to_image
from ..thirdparty.FBCNN import FBCNNProcessor
from ..utils.logger import setup_logger

__all__ = ["jpeg_restore_processor"]

llog = setup_logger(name=__name__)

class jpeg_restore_processor():
    def __init__(self, model_path=None, params=None, device=None):
        self.model_path = model_path if model_path is not None else os.path.join(os.environ["TORCH_HOME"], "fbcnn_color.pth")
        self.params = params if params is not None else {"auto_detect": True, "compression_level": 0, "tile_size": 1024, "overlap": 32}
        self.fbcnn = FBCNNProcessor(model_path=self.model_path, device=device)

        def process_func(iap_data):
            input_tensor = to_tensor(iap_data[0], mean=(0,0,0), std=(1.0, 1.0, 1.0))
            try:
                restored_tensor = self.fbcnn.process_image(input_tensor, **self.params)
                restored_img = to_image(restored_tensor, cvtcolor=True, valmin=0.0, valmax=1.0)
                return (restored_img, iap_data[1])
            except Exception as e:
                llog.warning(repr(e))
            return None

        self.process_func = process_func

    def dump_config(self):
        config = {"__name__": "jpeg_restore_processor", "method": self.method}
        config.update(self.params)
        return config

    def __call__(self, iap_data):
        return self.process_func(iap_data)