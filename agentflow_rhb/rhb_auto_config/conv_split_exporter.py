import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ConvSplitExport:
    node_index: int
    node_name: str
    split_id: str
    channel_start: int
    channel_end: int
    input_name: str
    output_name: str
    onnx_path: str
    input_channel_start: int = 0
    input_channel_end: int = 0


def _load_onnx():
    import onnx  # type: ignore
    from onnx import numpy_helper  # type: ignore

    return onnx, numpy_helper


def _shape_for_name(graph, name: str) -> List[int]:
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name != name:
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return []
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            else:
                shape.append(-1)
        return shape
    return []


def _value_info(onnx, name: str, shape: List[int]):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape if shape else None)


def export_conv_output_channel_splits(
    source_onnx: Path,
    node_index: int,
    output_dir: Path,
    chunk_channels: int = 8,
    include_following_relu: bool = False,
) -> List[ConvSplitExport]:
    onnx, numpy_helper = _load_onnx()
    model = onnx.load(str(source_onnx))
    graph = model.graph
    nodes = list(graph.node)
    conv = nodes[node_index]
    if conv.op_type != "Conv":
        raise ValueError(f"node {node_index} is {conv.op_type}, expected Conv")
    if include_following_relu:
        if node_index + 1 >= len(nodes) or nodes[node_index + 1].op_type != "Relu":
            raise ValueError("include_following_relu requires Conv immediately followed by Relu")
        relu = nodes[node_index + 1]
    else:
        relu = None

    initializer_by_name = {item.name: item for item in graph.initializer}
    weight_name = conv.input[1]
    bias_name = conv.input[2] if len(conv.input) > 2 else ""
    weight = numpy_helper.to_array(initializer_by_name[weight_name])
    bias = numpy_helper.to_array(initializer_by_name[bias_name]) if bias_name else None
    out_channels = int(weight.shape[0])
    input_name = conv.input[0]
    input_shape = _shape_for_name(graph, input_name)
    output_shape = _shape_for_name(graph, relu.output[0] if relu else conv.output[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    exports: List[ConvSplitExport] = []
    for start in range(0, out_channels, chunk_channels):
        end = min(start + chunk_channels, out_channels)
        split_id = f"node{node_index:03d}_oc{start:02d}_{end:02d}"
        split_weight_name = f"{weight_name}_{split_id}"
        split_bias_name = f"{bias_name}_{split_id}" if bias_name else ""
        split_output_name = f"{conv.output[0]}_{split_id}"
        final_output_name = f"{(relu.output[0] if relu else conv.output[0])}_{split_id}"

        split_weight = numpy_helper.from_array(weight[start:end].copy(), name=split_weight_name)
        split_initializers = [split_weight]
        split_inputs = [input_name, split_weight_name]
        if bias is not None:
            split_bias = numpy_helper.from_array(bias[start:end].copy(), name=split_bias_name)
            split_initializers.append(split_bias)
            split_inputs.append(split_bias_name)

        conv_attrs = []
        for attr in conv.attribute:
            if attr.name == "weight_ch_scales":
                values = list(onnx.helper.get_attribute_value(attr))[start:end]
                conv_attrs.append(onnx.helper.make_attribute(attr.name, values))
            else:
                conv_attrs.append(attr)
        split_conv = onnx.helper.make_node(
            "Conv",
            split_inputs,
            [split_output_name],
            name=f"{conv.name}_{split_id}",
            **{attr.name: onnx.helper.get_attribute_value(attr) for attr in conv_attrs},
        )
        split_nodes = [split_conv]
        if relu is not None:
            relu_attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in relu.attribute}
            split_nodes.append(
                onnx.helper.make_node(
                    "Relu",
                    [split_output_name],
                    [final_output_name],
                    name=f"{relu.name}_{split_id}",
                    **relu_attrs,
                )
            )

        split_shape = list(output_shape)
        if len(split_shape) >= 2:
            split_shape[1] = end - start
        subgraph = onnx.helper.make_graph(
            split_nodes,
            f"{Path(source_onnx).stem}_{split_id}",
            [_value_info(onnx, input_name, input_shape)],
            [_value_info(onnx, final_output_name, split_shape)],
            initializer=split_initializers,
        )
        submodel = onnx.helper.make_model(subgraph, producer_name="rhb_auto_config_conv_split")
        del submodel.opset_import[:]
        submodel.opset_import.extend(model.opset_import)
        submodel.ir_version = model.ir_version
        out_path = output_dir / f"{Path(source_onnx).stem}_{split_id}.onnx"
        onnx.save(submodel, str(out_path))
        exports.append(
            ConvSplitExport(
                node_index=node_index,
                node_name=conv.name,
                split_id=split_id,
                channel_start=start,
                channel_end=end,
                input_name=input_name,
                output_name=final_output_name,
                onnx_path=str(out_path),
            )
        )
    return exports


def export_conv_input_output_channel_splits(
    source_onnx: Path,
    node_index: int,
    output_dir: Path,
    output_chunk_channels: int = 8,
    input_chunk_channels: int = 8,
) -> List[ConvSplitExport]:
    onnx, numpy_helper = _load_onnx()
    model = onnx.load(str(source_onnx))
    graph = model.graph
    conv = list(graph.node)[node_index]
    if conv.op_type != "Conv":
        raise ValueError(f"node {node_index} is {conv.op_type}, expected Conv")
    if int(next((onnx.helper.get_attribute_value(attr) for attr in conv.attribute if attr.name == "group"), 1)) != 1:
        raise ValueError("input/output channel split currently supports group=1 only")

    initializer_by_name = {item.name: item for item in graph.initializer}
    weight_name = conv.input[1]
    weight = numpy_helper.to_array(initializer_by_name[weight_name])
    out_channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    input_name = conv.input[0]
    input_shape = _shape_for_name(graph, input_name)
    output_shape = _shape_for_name(graph, conv.output[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    exports: List[ConvSplitExport] = []
    for out_start in range(0, out_channels, output_chunk_channels):
        out_end = min(out_start + output_chunk_channels, out_channels)
        for in_start in range(0, in_channels, input_chunk_channels):
            in_end = min(in_start + input_chunk_channels, in_channels)
            split_id = f"node{node_index:03d}_oc{out_start:02d}_{out_end:02d}_ic{in_start:02d}_{in_end:02d}"
            split_weight_name = f"{weight_name}_{split_id}"
            final_output_name = f"{conv.output[0]}_{split_id}"
            split_input_name = f"{input_name}_{split_id}"
            split_weight = numpy_helper.from_array(weight[out_start:out_end, in_start:in_end].copy(), name=split_weight_name)

            conv_attrs = []
            for attr in conv.attribute:
                if attr.name == "weight_ch_scales":
                    values = list(onnx.helper.get_attribute_value(attr))[out_start:out_end]
                    conv_attrs.append(onnx.helper.make_attribute(attr.name, values))
                else:
                    conv_attrs.append(attr)
            split_conv = onnx.helper.make_node(
                "Conv",
                [split_input_name, split_weight_name],
                [final_output_name],
                name=f"{conv.name}_{split_id}",
                **{attr.name: onnx.helper.get_attribute_value(attr) for attr in conv_attrs},
            )
            split_input_shape = list(input_shape)
            if len(split_input_shape) >= 2:
                split_input_shape[1] = in_end - in_start
            split_output_shape = list(output_shape)
            if len(split_output_shape) >= 2:
                split_output_shape[1] = out_end - out_start
            subgraph = onnx.helper.make_graph(
                [split_conv],
                f"{Path(source_onnx).stem}_{split_id}",
                [_value_info(onnx, split_input_name, split_input_shape)],
                [_value_info(onnx, final_output_name, split_output_shape)],
                initializer=[split_weight],
            )
            submodel = onnx.helper.make_model(subgraph, producer_name="rhb_auto_config_conv_split")
            del submodel.opset_import[:]
            submodel.opset_import.extend(model.opset_import)
            submodel.ir_version = model.ir_version
            out_path = output_dir / f"{Path(source_onnx).stem}_{split_id}.onnx"
            onnx.save(submodel, str(out_path))
            exports.append(
                ConvSplitExport(
                    node_index=node_index,
                    node_name=conv.name,
                    split_id=split_id,
                    channel_start=out_start,
                    channel_end=out_end,
                    input_name=split_input_name,
                    output_name=final_output_name,
                    onnx_path=str(out_path),
                    input_channel_start=in_start,
                    input_channel_end=in_end,
                )
            )
    return exports


def export_stride_conv_im2col_1x1_splits(
    source_onnx: Path,
    node_index: int,
    output_dir: Path,
    output_chunk_channels: int = 8,
    flat_input_chunk_channels: int = 8,
) -> List[ConvSplitExport]:
    onnx, numpy_helper = _load_onnx()
    model = onnx.load(str(source_onnx))
    graph = model.graph
    conv = list(graph.node)[node_index]
    if conv.op_type != "Conv":
        raise ValueError(f"node {node_index} is {conv.op_type}, expected Conv")
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in conv.attribute}
    if list(attrs.get("kernel_shape", [])) != [3, 3]:
        raise ValueError("im2col exporter currently supports 3x3 Conv only")
    if int(attrs.get("group", 1)) != 1:
        raise ValueError("im2col exporter currently supports group=1 only")

    initializer_by_name = {item.name: item for item in graph.initializer}
    weight_name = conv.input[1]
    weight = numpy_helper.to_array(initializer_by_name[weight_name])
    out_channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    flat_weight = weight.reshape(out_channels, in_channels * 9, 1, 1)
    flat_channels = int(flat_weight.shape[1])
    output_shape = _shape_for_name(graph, conv.output[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    exports: List[ConvSplitExport] = []
    for out_start in range(0, out_channels, output_chunk_channels):
        out_end = min(out_start + output_chunk_channels, out_channels)
        for flat_start in range(0, flat_channels, flat_input_chunk_channels):
            flat_end = min(flat_start + flat_input_chunk_channels, flat_channels)
            split_id = f"node{node_index:03d}_im2col_oc{out_start:02d}_{out_end:02d}_fc{flat_start:03d}_{flat_end:03d}"
            split_input_name = f"{conv.input[0]}_im2col_{split_id}"
            split_output_name = f"{conv.output[0]}_{split_id}"
            split_weight_name = f"{weight_name}_im2col_{split_id}"
            split_weight = numpy_helper.from_array(flat_weight[out_start:out_end, flat_start:flat_end].copy(), name=split_weight_name)
            split_input_shape = [1, flat_end - flat_start, output_shape[2], output_shape[3]]
            split_output_shape = list(output_shape)
            split_output_shape[1] = out_end - out_start
            weight_scales = list(attrs.get("weight_ch_scales", []))[out_start:out_end]
            split_conv = onnx.helper.make_node(
                "Conv",
                [split_input_name, split_weight_name],
                [split_output_name],
                name=f"{conv.name}_{split_id}",
                dilations=[1, 1],
                group=1,
                kernel_shape=[1, 1],
                pads=[0, 0, 0, 0],
                strides=[1, 1],
                weight_ch_scales=weight_scales,
                weight_bitdepth=attrs.get("weight_bitdepth", 8),
                input_scale=attrs.get("input_scale", 1024.0),
                input_bitdepth=attrs.get("input_bitdepth", 8),
                output_scale=attrs.get("output_scale", 1024.0),
                output_bitdepth=attrs.get("output_bitdepth", 8),
            )
            subgraph = onnx.helper.make_graph(
                [split_conv],
                f"{Path(source_onnx).stem}_{split_id}",
                [_value_info(onnx, split_input_name, split_input_shape)],
                [_value_info(onnx, split_output_name, split_output_shape)],
                initializer=[split_weight],
            )
            submodel = onnx.helper.make_model(subgraph, producer_name="rhb_auto_config_conv_split")
            del submodel.opset_import[:]
            submodel.opset_import.extend(model.opset_import)
            submodel.ir_version = model.ir_version
            out_path = output_dir / f"{Path(source_onnx).stem}_{split_id}.onnx"
            onnx.save(submodel, str(out_path))
            exports.append(
                ConvSplitExport(
                    node_index=node_index,
                    node_name=conv.name,
                    split_id=split_id,
                    channel_start=out_start,
                    channel_end=out_end,
                    input_name=split_input_name,
                    output_name=split_output_name,
                    onnx_path=str(out_path),
                    input_channel_start=flat_start,
                    input_channel_end=flat_end,
                )
            )
    return exports


def export_stride_conv_offset_1x1_splits(
    source_onnx: Path,
    node_index: int,
    output_dir: Path,
    output_chunk_channels: int = 8,
    input_chunk_channels: int = 8,
) -> List[ConvSplitExport]:
    onnx, numpy_helper = _load_onnx()
    model = onnx.load(str(source_onnx))
    graph = model.graph
    conv = list(graph.node)[node_index]
    if conv.op_type != "Conv":
        raise ValueError(f"node {node_index} is {conv.op_type}, expected Conv")
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in conv.attribute}
    kernel = list(attrs.get("kernel_shape", []))
    if len(kernel) != 2:
        raise ValueError("offset exporter currently supports 2D Conv only")
    if int(attrs.get("group", 1)) != 1:
        raise ValueError("offset exporter currently supports group=1 only")

    initializer_by_name = {item.name: item for item in graph.initializer}
    weight_name = conv.input[1]
    weight = numpy_helper.to_array(initializer_by_name[weight_name])
    out_channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    output_shape = _shape_for_name(graph, conv.output[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    exports: List[ConvSplitExport] = []
    for kh in range(kernel[0]):
        for kw in range(kernel[1]):
            for out_start in range(0, out_channels, output_chunk_channels):
                out_end = min(out_start + output_chunk_channels, out_channels)
                for in_start in range(0, in_channels, input_chunk_channels):
                    in_end = min(in_start + input_chunk_channels, in_channels)
                    split_id = (
                        f"node{node_index:03d}_off{kh}{kw}_"
                        f"oc{out_start:02d}_{out_end:02d}_ic{in_start:02d}_{in_end:02d}"
                    )
                    split_input_name = f"{conv.input[0]}_offset{kh}{kw}_{split_id}"
                    split_output_name = f"{conv.output[0]}_{split_id}"
                    split_weight_name = f"{weight_name}_offset{kh}{kw}_{split_id}"
                    split_weight = numpy_helper.from_array(
                        weight[out_start:out_end, in_start:in_end, kh, kw]
                        .reshape(out_end - out_start, in_end - in_start, 1, 1)
                        .copy(),
                        name=split_weight_name,
                    )
                    split_input_shape = [1, in_end - in_start, output_shape[2], output_shape[3]]
                    split_output_shape = list(output_shape)
                    split_output_shape[1] = out_end - out_start
                    weight_scales = list(attrs.get("weight_ch_scales", []))[out_start:out_end]
                    split_conv = onnx.helper.make_node(
                        "Conv",
                        [split_input_name, split_weight_name],
                        [split_output_name],
                        name=f"{conv.name}_{split_id}",
                        dilations=[1, 1],
                        group=1,
                        kernel_shape=[1, 1],
                        pads=[0, 0, 0, 0],
                        strides=[1, 1],
                        weight_ch_scales=weight_scales,
                        weight_bitdepth=attrs.get("weight_bitdepth", 8),
                        input_scale=attrs.get("input_scale", 1024.0),
                        input_bitdepth=attrs.get("input_bitdepth", 8),
                        output_scale=attrs.get("output_scale", 1024.0),
                        output_bitdepth=attrs.get("output_bitdepth", 8),
                    )
                    subgraph = onnx.helper.make_graph(
                        [split_conv],
                        f"{Path(source_onnx).stem}_{split_id}",
                        [_value_info(onnx, split_input_name, split_input_shape)],
                        [_value_info(onnx, split_output_name, split_output_shape)],
                        initializer=[split_weight],
                    )
                    submodel = onnx.helper.make_model(subgraph, producer_name="rhb_auto_config_conv_split")
                    del submodel.opset_import[:]
                    submodel.opset_import.extend(model.opset_import)
                    submodel.ir_version = model.ir_version
                    out_path = output_dir / f"{Path(source_onnx).stem}_{split_id}.onnx"
                    onnx.save(submodel, str(out_path))
                    exports.append(
                        ConvSplitExport(
                            node_index=node_index,
                            node_name=conv.name,
                            split_id=split_id,
                            channel_start=out_start,
                            channel_end=out_end,
                            input_name=split_input_name,
                            output_name=split_output_name,
                            onnx_path=str(out_path),
                            input_channel_start=in_start,
                            input_channel_end=in_end,
                        )
                    )
    return exports


def save_conv_split_exports(exports: List[ConvSplitExport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in exports], indent=2), encoding="utf-8")


def render_conv_split_exports(exports: List[ConvSplitExport]) -> str:
    lines = ["split_id\tnode_index\tchannels\tinput\toutput\tonnx_path"]
    for item in exports:
        input_span = ""
        if item.input_channel_end > item.input_channel_start:
            input_span = f"\tic={item.input_channel_start}:{item.input_channel_end}"
        lines.append(
            f"{item.split_id}\t{item.node_index}\t{item.channel_start}:{item.channel_end}{input_span}\t"
            f"{item.input_name}\t{item.output_name}\t{item.onnx_path}"
        )
    return "\n".join(lines)
