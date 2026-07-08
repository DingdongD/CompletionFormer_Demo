from ._ckpt_hw128_common import Dec2BlockConv1Ckpt

input_layouts = ["CHW"]
ifmap_sz = [[32, 128, 128]]
op_version = 18
batch_size = 1
tensor_16 = [".bias"]


class Model(Dec2BlockConv1Ckpt):
    pass
