
import numpy as np
import sys, re, yaml, onnx

# add python path ./acsim
sys.path.append("./acsim")
import ACONNXRUNTime as rt

# load test cases from test_cases.yaml
with open("test_cases.yml", 'r') as stream:
    try:
        test_cases = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

output_root = "./output"
test_case = []

if len(sys.argv) > 1:
    output_root = sys.argv[1]
if len(sys.argv) > 2:
    test_case = sys.argv[2].split(",")

org_stdout = sys.stdout
test_cases = test_cases["tests"]
case_dict = {}
for case in test_cases:
    case_dict[int(case["id"])] = case
results_dict = {}
results_dict["pass"] = []
results_dict["fail"] = []
results_dict["pass with tolerance"] = []
results_dict["not found"] = []
test_ids = range(70)
if len(test_case) != 0:
    # find the test case id that with the test_case name
    test_ids = []
    for case in test_cases:
        if case["model"] in test_case:
            test_ids.append(int(case["id"]))
            

for case_id in test_ids: #[51, 52, 43, 44]
    if case_id not in case_dict:
        print ("case_id:", case_id, "not exist")
        continue
    test_case = case_dict[case_id]
    model = test_case["model"]
    tolerance = 1
    if "tolerance" in test_case:
        tolerance = test_case["tolerance"]
    input_names = []
    if "input_names" in test_case:
        input_names = test_case["input_names"]
    if "out_scale_fixed" in test_case:
        out_scale_fixed = test_case["out_scale_fixed"]
        out_scale_fixed = eval(out_scale_fixed)
    else:    
        out_scale_fixed = 0
    if "input_scales" in test_case:
        input_scales = test_case["input_scales"]
    else:
        input_scales = []

    print ("model:", model)
    model_path = 'onnx_models/' + model + '.onnx'
    mlir_path = output_root + "/" + model + "/" + model + ".tmp"
    # check if the model exists
    try:
        with open(model_path, 'r') as f:
            pass
    except:
        results_dict["not found"].append((case_id, model))
        continue
    try:
        with open(mlir_path, 'r') as f:
            pass
    except:
        results_dict["not found"].append((case_id, model))
        continue

    # Load the model as a graph
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = rt.InferenceSession(
        model_path, providers=rt.get_available_providers(), sess_options=sess_options)

    # redircet stdout to file
    print (output_root + "/" + model + "/onnx_output.txt")
    sys.stdout = open(output_root + "/" + model + "/onnx_output.txt", "w")  
    # get the line has "achigh.Unstick"
    # with open(mlir_path, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if "achigh.Unstick" in line and out_scale_fixed == 0:
    #             #get number after "qScale = "
    #             out_scale_fixed = float(re.findall(r"qScale\ =\ [0-9]+", line)[0]\
    #                                 .replace("qScale = ", ""))
    #             print (out_scale_fixed)
    #             break
    if out_scale_fixed == 0:
        # load the onnx model and get the node which is the last node
        onnx_model = onnx.load(model_path)
        # get the node which is the last node
        model_output = sess.get_outputs()
        for i in range(len(model_output)):
            for node in onnx_model.graph.node:
                if model_output[i].name == node.output[0]:
                    for attr in node.attribute:
                        if attr.name == "output_scale":
                            out_scale_fixed = float(attr.f)
                            print ("out_scale_fixed:", out_scale_fixed)
                            break
    diff_num = 0
    diff_max = 0
    output_num = 0
    input_order = []
    for seed in ["1", "33"]:
        if len(results_dict["not found"]) > 0 and results_dict["not found"][-1] == (case_id, model):
            continue
        csim_path = output_root + "/" + model + '/' + model + '_' + seed + '/'
        net_input_loc = csim_path + "net_csim_input.npz"
        net_output_loc = csim_path + "net_csim_output.npz"
        try:
            with open(net_input_loc, 'r') as f:
                pass
        except:
            results_dict["not found"].append((case_id, model))
            continue
        try:
            with open(net_output_loc, 'r') as f:
                pass
        except:
            results_dict["not found"].append((case_id, model))
            continue
        # load csim results
        csim_output = np.load(csim_path + "net_csim_output.npz")
        csim_input = np.load(csim_path + "net_csim_input.npz")
        if len(input_names) == 0:
            input_names = list(csim_input.keys())
        if "input_order" in test_case and input_order == []:
            input_order = test_case["input_order"]
            assert len(input_order) == len(input_names), "input_order and input_names should have the same length, but got {} and {}".format(len(input_order), len(input_names))
            input_names = [input_names[input_order[i]] for i in range(len(input_order))]
        print ("input names:", input_names)
        output_names = list(csim_output.keys())
        csim_out = csim_output[output_names[0]].astype(np.float32)
        # get the input name and shape for the model

        input_dict = {}
        for idx, key in enumerate(input_names):
            input_name = sess.get_inputs()[idx].name
            input_shape = sess.get_inputs()[idx].shape
            print ("onnx model input:", input_name, input_shape)
            inputs = csim_input[key].astype(np.float32) 
            if len(input_scales) > 0:
                inputs = inputs / input_scales[idx]
            print (inputs.shape)
            input_data = np.zeros(input_shape, dtype=np.float32)
            if (len(input_shape) == 4):
                input_data[:input_shape[0], 
                           :input_shape[1], 
                           :input_shape[2], 
                           :input_shape[3]] = inputs[:input_shape[0], 
                                                     :input_shape[1], 
                                                     :input_shape[2], 
                                                     :input_shape[3]] # (np.random.random_sample(input_shape).astype(np.float32) * 128).clip(-128, 127)
            if (len(input_shape) == 3):
                input_data[:input_shape[0], 
                           :input_shape[1], 
                           :input_shape[2]] = inputs[0, 
                                                     :input_shape[0], 
                                                     :input_shape[1], 
                                                     :input_shape[2]]
            input_dict[input_name] = input_data
            # print (input_data)
        output = sess.run(None, input_dict)
        print ("output shape:", output[0].shape, csim_out.shape)
        # out_ref = output[0][0,:10,:,:]
        slices = tuple(slice(0, dim) for dim in output[0].shape)
        output_num = output_num + np.prod(output[0].shape)
        if len(output[0].shape) == 3:
            slices = (slice(0, 1),) + slices  
        # csim_ref = csim_out[0,:10,:,:]
        # find first non-zero element in the output
        idx = np.where(output[0].flatten() != 0)[0][0]
        if out_scale_fixed == 0 and output[0].flatten()[idx] != 0:
            output_scale = csim_out[slices].flatten()[idx] / output[0].flatten()[idx]
        else:
            output_scale = out_scale_fixed
        print ("output scale:", output_scale)
        onnx_output = output[0] #[:,1:,:,:]#dmu_test0
        np.savez(output_root + "/" + model + "/onnx_output.npz", rt=output_scale * onnx_output, csim=csim_out)
        print ("output:", output_scale * onnx_output.flatten()[:10])
        print ("csim output:", csim_out[slices].flatten()[:10])
        # print ("output:\n", output_scale * onnx_output[0, :10,:10])
        # print ("csim output:\n", csim_out[slices][0, 0,:10,:10])
        print ("max diff:", np.max(np.abs(output_scale * onnx_output - csim_out[slices])))
        diff_max = max(diff_max, np.max(np.abs(output_scale * onnx_output - csim_out[slices])))

        # find the where the difference is 1
        np.set_printoptions(threshold=np.inf)
        diff_pos = np.where(np.abs(output_scale * onnx_output - csim_out[slices]) >= 1)
        # sort diff_pos by the diff value
        # if len(onnx_output.shape) == 4:
        #     diff_pos = [diff_pos[0][np.argsort(-np.abs(output_scale * onnx_output[diff_pos] - csim_out[slices][diff_pos]))],
        #                 diff_pos[1][np.argsort(-np.abs(output_scale * onnx_output[diff_pos] - csim_out[slices][diff_pos]))],
        #                 diff_pos[2][np.argsort(-np.abs(output_scale * onnx_output[diff_pos] - csim_out[slices][diff_pos]))],
        #                 diff_pos[3][np.argsort(-np.abs(output_scale * onnx_output[diff_pos] - csim_out[slices][diff_pos]))]]
        # elif len(onnx_output.shape) == 3:
        #     diff_pos = [diff_pos[0][np.argsort(-np.abs(output_scale * onnx_output[diff_pos[:3]] - csim_out[slices][diff_pos]))],
        #                 diff_pos[1][np.argsort(-np.abs(output_scale * onnx_output[diff_pos[:3]] - csim_out[slices][diff_pos]))],
        #                 diff_pos[2][np.argsort(-np.abs(output_scale * onnx_output[diff_pos[:3]] - csim_out[slices][diff_pos]))],
        #                 diff_pos[3][np.argsort(-np.abs(output_scale * onnx_output[diff_pos[:3]] - csim_out[slices][diff_pos]))]]
        print ("diff num:", len(diff_pos[0]))
        diff_num = diff_num + len(diff_pos[0])
        shown_num = 0
        for i in range(len(diff_pos[0])):
            if  shown_num > 3:
                print ("...")
                break
            try:
                # get the flatten index of the diff
                flatten_idx = diff_pos[0][i] * onnx_output.shape[1] * onnx_output.shape[2] * onnx_output.shape[3] + \
                                diff_pos[1][i] * onnx_output.shape[2] * onnx_output.shape[3] + \
                                diff_pos[2][i] * onnx_output.shape[3] + diff_pos[3][i]
                # print ("flatten idx:", flatten_idx) 
                diff_value = output_scale * onnx_output[diff_pos[0][i], diff_pos[1][i], diff_pos[2][i], diff_pos[3][i]], csim_out[diff_pos[0][i], diff_pos[1][i], diff_pos[2][i], diff_pos[3][i]]
                
            except:
                try:
                    # get the flatten index of the diff
                    flatten_idx = diff_pos[1][i] * onnx_output.shape[1] * onnx_output.shape[2] + \
                                    diff_pos[2][i] * onnx_output.shape[2] + diff_pos[3][i]
                    # print ("flatten idx:", flatten_idx) 
                    diff_value = output_scale * onnx_output.flatten()[flatten_idx], csim_out[slices].flatten()[flatten_idx]
                    
                    # print ("diff value:", output_scale * onnx_output.flatten()[flatten_idx], csim_out[slices].flatten()[flatten_idx])
                except:
                    continue
            if np.abs(diff_value[0] - diff_value[1]) > 2:
                shown_num += 1
                print ("diff pos:", diff_pos[0][i], diff_pos[1][i], diff_pos[2][i], diff_pos[3][i],)
            
                print ("flatten idx:", flatten_idx) 
                print ("diff value:", diff_value)
        # print ("where diff:", np.where(np.abs(output_scale * output[0] - csim_out[slices]) >= 1))
        # print ("diff:", out_ref/csim_ref)
        # print (output_scale * output[0])
    sys.stdout.close()
    sys.stdout = org_stdout
    print (output_root + "/" + model + "/onnx_output.txt")
    if diff_max == 0:
        print ("[pass]")
        if "pass" in results_dict:
            results_dict["pass"].append((case_id, model))
    elif diff_max <= tolerance:
        print ("max diff:", diff_max, "diff num:", diff_num, "/", output_num)
        print ("[pass with tolerance]")
        results_dict["pass with tolerance"].append((case_id, model))
    else:
        print ("[fail]")
        results_dict["fail"].append((case_id, model))
print ("results:", results_dict)
print ("pass:", len(results_dict["pass"]))
print ("pass with tolerance:", len(results_dict["pass with tolerance"]))
print ("fail:", len(results_dict["fail"]), results_dict["fail"])
print ("not found:", len(results_dict["not found"]), results_dict["not found"])
print ("done")
