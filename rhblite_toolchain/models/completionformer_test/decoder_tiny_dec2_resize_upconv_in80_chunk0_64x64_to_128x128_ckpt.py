from ._ckpt_hw128_common import Dec2ResizeUpConvChunkCkpt

input_layouts = ["CHW"]
ifmap_sz = [[80, 64, 64]]
op_version = 18
batch_size = 1
tensor_16 = [".bias"]


class Model(Dec2ResizeUpConvChunkCkpt):
    def __init__(self):
        super().__init__(0, True)
