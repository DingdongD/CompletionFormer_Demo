##rhb_test/cv_onnx.py
import onnx, math
import onnxruntime as rt
from onnxsim import simplify
#from models.test1 import Model
import torch
import torch.onnx
import numpy as np
from util.util import optimize_slopes_scale
import random
import argparse
from importlib import import_module
from util.spu import get_opt_exp_scale
from util.quant_onnx import quant_onnx
from onnx import shape_inference

softmax_mode = 0
kv_token_len = 2
token_num = 1


#import from argparse
parser = argparse.ArgumentParser(description='To ONNX')
parser.add_argument('td', default="", type=str)
parser.add_argument('--masked_attention', default=False, type=bool)
parser.add_argument('--quant', dest='quant', action='store_true')
parser.add_argument('--no-quant', dest='quant', action='store_false')
parser.add_argument('--op_version', default=18, type=int)
parser.set_defaults(quant=True)
args = parser.parse_args()
module = import_module("models." + args.td)
tensor_16 = []
try:
    tensor_16 = module.tensor_16
except:
    pass
batch_onnx = False
try:
    batch_onnx = module.batch_onnx
except:
    pass
tensor_flt = []
try:
    tensor_flt = module.tensor_flt
except:
    pass
tensor_4 = []
try:
    tensor_4 = module.tensor_4
except:
    pass
try:
    args.op_version = module.op_version
except:
    pass

# if not want model to change weight each time you run,
# add the follow three lines
torch.manual_seed(100)
random.seed(100)
np.random.seed(100)

model = module.Model()
ifmap_sz = module.ifmap_sz
# convert model to onnx
#dummy_input = torch.randn(1, 3, 21, 21)
dummy_input = tuple()
dummy_input_bat = tuple()
names = tuple()
for idx, sz in enumerate(ifmap_sz):
    dummy_input = dummy_input + (torch.randn((1,) + tuple(sz)),)
    dummy_input_bat = dummy_input_bat + (torch.randn((13,) + tuple(sz)),)
    names = names + ("input" + str(idx),)
print (model(*dummy_input))
torch.onnx.export(model, dummy_input, args.td + "_org.onnx", verbose=True, opset_version=args.op_version, input_names=names)
print (model(*dummy_input_bat))
torch.onnx.export(model, dummy_input_bat, args.td + "_batch.onnx", verbose=True, opset_version=args.op_version, input_names=names)
# infer shape and save

if not batch_onnx:
    model = onnx.load(args.td + "_org.onnx")
else:
    model = onnx.load(args.td + "_batch.onnx")
try:
    model_simp, check = simplify(model)
    onnx.save(model_simp, args.td + "_simp.onnx")
except:
    print ("simplify failed")
    onnx.save(model, args.td + "_simp.onnx")
batch_size = 1

try:
    module = import_module("models." + args.td)
    batch_size = module.batch_size
except:
    pass
print ("save to", args.td + "_simp.onnx")

quant_onnx(args.td + "_simp", tensor_16, tensor_4, tensor_flt, batch_size=batch_size)
