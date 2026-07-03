from ._ckpt_hw128_common import HeadConvCkpt

input_layouts = ["CHW"]
ifmap_sz = [[96, 128, 128]]
op_version = 18
batch_size = 1
tensor_16 = [".bias"]


class Model(HeadConvCkpt):
    def __init__(self):
        super().__init__("backbone.dep_dec0", 96, 1, relu=True)
