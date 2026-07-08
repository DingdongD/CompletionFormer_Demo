from ACompiler import ACompiler
import argparse
import graphviz
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument("--arch_path", type=str, default="arch_duo_16.yaml")
parser.add_argument("--model", type=str, default="models/attn.onnx")
parser.add_argument("--model_py", type=str, default="models/test/attn.py")
parser.add_argument("--output_path", type=str, default="output/attn")
parser.add_argument("--log_path", type=str, default="")
parser.add_argument("--opt_level", type=int, default=3)
parser.add_argument("--split", type=int, default=16)
parser.add_argument("--layouts", type=str, default=r'onnx::MatMul_0=BWC')
parser.add_argument('-c', '--codegen', help='enable codeGen', type=int, default=0)
parser.add_argument('-n', '--max_concat_in_cnt', help='ma concat buffer number', type=int, default=100)
parser.add_argument('-s', '--sim', help='enable dummy simulation', type=int, default=0)
parser.add_argument('-a', '--addr', help='addr allocation policy', type=int, default=0)
parser.add_argument('-b', '--bypass_dmu', help='bypass dmu', type=int, default=0)   
parser.add_argument('-l', '--load_entire_rope', help='load entire rope params', type=int, default=0)  
parser.add_argument("--token_index_range", type=str, default=r'0:0') 
args = parser.parse_args()

try:
    module = import_module(args.model_py)
    args.token_index_range = module.token_index_range
except:
    pass

try:
    module = import_module(args.model_py)
    args.addr = module.addr_policy
except:
    pass

compiler = ACompiler(arch_path=args.arch_path, is_dummy_profiler=(args.sim == 1), l2_size=100, ctc_num=1)

# parse memory_img=BWC,flatten_pos_emb=BWC,ca_qpos_sine=WBC,ca_text=WBC,ref_point_proj=WBC into a dict of {memory_img: BWC, flatten_pos_emb: BWC, ...}
layouts = {}
if len(args.layouts) == 0:
# from model_py import input_layouts

    

    module = import_module(args.model_py)
    for idx, input_layout in enumerate(module.input_layouts):
        layouts["input" + str(idx)] = "B" + input_layout
else:    
    for layout in args.layouts.split(","):
        if len(layout.split("=")) == 2:
            layouts[layout.split("=")[0]] = layout.split("=")[1]
print (layouts)
try: 
    compiler.compile(model=args.model,
                    layout=layouts,
                    output_path=args.output_path,
                    log_path=args.log_path,
                    opt_level=args.opt_level,
                    fast_mapping=True,
                    generate_ddr=args.codegen,
                    split_granularity=args.split,
                    max_concat_in_cnt=args.max_concat_in_cnt,
                    verbose_output=True,
                    skip_layout=False,
                    bypass_dmu=args.bypass_dmu > 0,
                    addr_alloc_policy=args.addr,
                    load_entire_rope=args.load_entire_rope > 0,
                    token_index_range=args.token_index_range)
except Exception as e:
    print (e)

