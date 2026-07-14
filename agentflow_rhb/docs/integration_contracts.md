# Integration Contracts

This file defines the interfaces needed to turn the current skeleton into a fully automatic deployment loop.

## Graph Importer Contract

Input:

```text
model path
input specs
checkpoint path
representative samples
```

Current adapter:

```bash
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py import-onnx --onnx onnx_models/<model>.onnx
```

Output:

```json
{
  "nodes": [
    {
      "id": "string",
      "op": "Conv2d",
      "shape_in": [[1, 96, 128, 128]],
      "shape_out": [[1, 32, 128, 128]],
      "attrs": {"kernel": [3, 3], "padding": [1, 1]},
      "params_bytes_fp32": 110592
    }
  ],
  "edges": []
}
```

## Compiler Adapter Contract

Input:

```json
{
  "submodel_name": "string",
  "model_file": "path",
  "layout": "input0=BWC",
  "output_dir": "path"
}
```

Current adapter:

```bash
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py compile-cmodel \
  --model completionformer_test.<submodel> \
  --layout "input0=BWC"
```

It wraps:

```text
make model=<model> cv_model
make model=<model> compile layout=<layout>
make model=<model> seed=1 cmodel
```

Output:

```json
{
  "compile_status": "pass|fail",
  "cmodel_status": "pass|fail",
  "packed_weight_bytes": 123456,
  "logs": ["path"]
}
```

## Board Adapter Contract

Input:

```json
{
  "packer_dir": "path",
  "submodel_name": "string",
  "inputs": ["path"],
  "clear_wr_done_before_run": true
}
```

Current adapters:

```bash
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py pack \
  --compile-output-dir <compiled-model-dir> \
  --packer-output-dir <packer-dir>

python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py board-run \
  --packer-dir <packer-dir>
```

`board-run` can call the default board `deploy.py` or a custom runner copied from Host using `--local-runner`.

Output:

```json
{
  "board_status": "pass|fail_timeout|fail_accuracy|fail_runtime",
  "latency_ms": 104.2,
  "counters": {
    "sta_npu_opt_cnt": 46136
  },
  "outputs": ["path"],
  "log": "path"
}
```

## Rule Feedback Contract

Every failure should produce a candidate rule update:

```json
{
  "pattern": "Sigmoid",
  "compile_status": "miscompiled",
  "board_status": "fail_accuracy",
  "decision": "host",
  "evidence": "Conv+Sigmoid lowered to HardSwish for cf_dec0"
}
```

The rule update should be reviewed before becoming a stable rule.
