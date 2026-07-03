from ._ckpt_hw128_common import ResizeBlockCkpt

input_layouts = ["CHW"]
ifmap_sz = [[72, 32, 32]]
op_version = 18
batch_size = 1
tensor_16 = [".bias"]


class Model(ResizeBlockCkpt):
    def __init__(self):
        super().__init__("backbone.dec3", 72, 32, (64, 64))
