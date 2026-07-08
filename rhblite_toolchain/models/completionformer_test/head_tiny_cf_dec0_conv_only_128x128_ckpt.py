from ._ckpt_hw128_common import HeadConvCkpt

input_layouts = ["CHW"]
ifmap_sz = [[80, 128, 128]]
op_version = 18
batch_size = 1
tensor_16 = [".bias"]


class Model(HeadConvCkpt):
    def __init__(self):
        super().__init__("backbone.cf_dec0", 80, 1, sigmoid=False)
