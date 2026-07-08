from ._ckpt_hw128_common import ResizeBlockCkpt

input_layouts = ["CHW"]
ifmap_sz = [[112, 16, 16]]
op_version = 18
batch_size = 1
tensor_16 = [".bias"]


class Model(ResizeBlockCkpt):
    def __init__(self):
        super().__init__("backbone.dec4", 112, 48, (32, 32))
