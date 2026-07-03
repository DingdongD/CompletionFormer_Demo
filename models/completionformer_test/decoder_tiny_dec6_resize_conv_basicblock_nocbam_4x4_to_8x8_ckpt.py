from ._ckpt_hw128_common import ResizeBlockCkpt

input_layouts = ["CHW"]
ifmap_sz = [[192, 4, 4]]
op_version = 18
batch_size = 1
tensor_16 = [".bias"]


class Model(ResizeBlockCkpt):
    def __init__(self):
        super().__init__("backbone.dec6", 192, 96, (8, 8))
