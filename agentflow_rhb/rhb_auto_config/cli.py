import argparse
import re
from pathlib import Path

from annotator import annotate_graph, render_annotations, save_annotations_json
from artifact_manifest import build_artifact_manifest, render_artifact_manifest, save_artifact_manifest
from backlog_executor import execute_backlog, render_backlog_execution, save_backlog_execution
from board_runner import BoardConfig, render_board_result, run_board
from case_generator import generate_case, render_generated_case, save_generated_case
from compile_runner import CompileConfig, render_compile_result, run_compile_cmodel
from conv_split_exporter import (
    export_conv_input_output_channel_splits,
    export_conv_output_channel_splits,
    export_stride_conv_im2col_1x1_splits,
)
from cost_calibrator import calibrate_costs, render_cost_calibration, save_cost_calibration
from deep_search import render_deep_search_result, run_deep_search, save_deep_search_result
from deployment_package import generate_deployment_package, render_deployment_package
from failure_localizer import localize_failure, render_failure_localization, save_failure_localization
from feedback import render_rule_update, suggest_rule_from_result
from graph_importer import import_onnx, render_onnx_summary, save_onnx_json
from health_report import build_health_report, render_health_report, save_health_report_json
from inventory import render_inventory, scan_models
from layout_model import analyze_layout, render_layout_report, save_layout_report_json
from package_contract import build_package_contract, render_package_contract, save_package_contract
from package_validator import render_package_validation, save_package_validation, validate_package
from packer_runner import PackConfig, run_pack
from pipeline import DeployLoopConfig, render_deploy_loop, run_deploy_loop
from outlier_analysis import analyze_npz_pairs, load_boundary_csv, render_outlier_report, save_outlier_json
from planner import load_case, render_plan
from portfolio import render_portfolio, save_portfolio_json, score_onnx_portfolio
from production_loop import build_production_plan, render_production_plan, save_production_plan
from quant_diagnostics import analyze_npz, render_quant_diagnostics, save_quant_diagnostics
from remote_training import (
    build_remote_training_plan,
    fetch_remote_training,
    load_remote_training_profile,
    render_remote_training_plan,
    save_remote_training_plan,
    status_remote_training,
    stop_remote_training,
    submit_remote_training,
)
from deployment_graph import build_deployment_graph, render_deployment_graph, save_deployment_graph_json
from regions import build_region_plan, load_effective_budget, render_region_plan, save_region_plan_json
from retry_planner import plan_retries, render_retry_plan, save_retry_plan
from retry_executor import execute_retry_plan, render_retry_execution, save_retry_execution
from rule_db import RuleDB
from schema import DEFAULT_DEPLOYMENT_POLICY, DEFAULT_REMOTE_TRAINING_CONFIG, DEFAULT_REPORT_DIR, DEFAULT_RULE_DB
from source_profiler import profile_source_tree, render_source_profile, save_source_profile
from submodel_exporter import export_rhb_submodels, render_exported_submodels
from validation_runner import render_validation_report, save_validation_report, validate_models
from rewrite_backlog import build_rewrite_backlog, render_rewrite_backlog, save_rewrite_backlog_json
from split_contract import build_conv_split_contract, render_split_contract, save_split_contract


def write_report(name: str, text: str) -> Path:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = DEFAULT_REPORT_DIR / name
    path.write_text(text + "\n", encoding="utf-8")
    return path


def cmd_summarize_rules(args: argparse.Namespace) -> None:
    rules = RuleDB.load(Path(args.rules))
    text = "\n".join(rules.summary_lines())
    path = write_report("rule_summary.txt", text)
    print(text)
    print(f"REPORT: {path}")


def cmd_scan_models(args: argparse.Namespace) -> None:
    files = scan_models(Path(args.models_root))
    text = render_inventory(files, limit=args.limit)
    path = write_report("model_inventory.txt", text)
    print(text)
    print(f"REPORT: {path}")


def cmd_plan(args: argparse.Namespace) -> None:
    rules = RuleDB.load(Path(args.rules))
    case = load_case(Path(args.case))
    text = render_plan(case, rules)
    path = write_report(f"plan_{case.case_name}.md", text)
    print(text)
    print(f"REPORT: {path}")


def cmd_import_onnx(args: argparse.Namespace) -> None:
    info = import_onnx(Path(args.onnx))
    text = render_onnx_summary(info, node_limit=args.node_limit)
    stem = Path(args.onnx).stem
    report = write_report(f"onnx_summary_{stem}.txt", text)
    json_path = DEFAULT_REPORT_DIR / f"onnx_graph_{stem}.json"
    save_onnx_json(info, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_annotate_onnx(args: argparse.Namespace) -> None:
    info = import_onnx(Path(args.onnx))
    rules = RuleDB.load(Path(args.rules))
    annotations = annotate_graph(info, rules)
    text = render_annotations(annotations)
    stem = Path(args.onnx).stem
    report = write_report(f"onnx_annotation_{stem}.tsv", text)
    json_path = DEFAULT_REPORT_DIR / f"onnx_annotation_{stem}.json"
    save_annotations_json(annotations, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_analyze_layout(args: argparse.Namespace) -> None:
    info = import_onnx(Path(args.onnx))
    report_obj = analyze_layout(info, data_width_bits=args.data_width_bits)
    text = render_layout_report(report_obj, limit=args.limit)
    stem = Path(args.onnx).stem
    report = write_report(f"layout_analysis_{stem}.md", text)
    json_path = DEFAULT_REPORT_DIR / f"layout_analysis_{stem}.json"
    save_layout_report_json(report_obj, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_optimize_onnx(args: argparse.Namespace) -> None:
    info = import_onnx(Path(args.onnx))
    rules = RuleDB.load(Path(args.rules))
    annotations = annotate_graph(info, rules)
    layout_report = analyze_layout(info, data_width_bits=args.data_width_bits)
    budget = int(args.effective_budget_bytes) if args.effective_budget_bytes else load_effective_budget()
    region_plan = build_region_plan(
        info,
        annotations,
        effective_weight_budget_bytes=budget,
        allow_approx_rewrites=args.allow_approx_rewrites,
    )
    deployment_graph = build_deployment_graph(region_plan)

    stem = Path(args.onnx).stem
    summary_path = write_report(f"onnx_summary_{stem}.txt", render_onnx_summary(info, node_limit=args.node_limit))
    graph_json = DEFAULT_REPORT_DIR / f"onnx_graph_{stem}.json"
    save_onnx_json(info, graph_json)

    annotation_path = write_report(f"onnx_annotation_{stem}.tsv", render_annotations(annotations))
    annotation_json = DEFAULT_REPORT_DIR / f"onnx_annotation_{stem}.json"
    save_annotations_json(annotations, annotation_json)

    layout_path = write_report(f"layout_analysis_{stem}.md", render_layout_report(layout_report, limit=args.node_limit))
    layout_json = DEFAULT_REPORT_DIR / f"layout_analysis_{stem}.json"
    save_layout_report_json(layout_report, layout_json)

    region_path = write_report(f"region_plan_{stem}.md", render_region_plan(region_plan))
    region_json = DEFAULT_REPORT_DIR / f"region_plan_{stem}.json"
    save_region_plan_json(region_plan, region_json)

    deployment_text = render_deployment_graph(deployment_graph)
    deployment_path = write_report(f"deployment_graph_{stem}.md", deployment_text)
    deployment_json = DEFAULT_REPORT_DIR / f"deployment_graph_{stem}.json"
    save_deployment_graph_json(deployment_graph, deployment_json)

    export_path = None
    if args.export_submodels:
        export_dir = Path(args.export_dir) if args.export_dir else DEFAULT_REPORT_DIR.parent / "work" / "submodels" / stem
        exported = export_rhb_submodels(region_json, export_dir, split_multi_output=args.split_multi_output)
        export_path = write_report(f"exported_submodels_{stem}.tsv", render_exported_submodels(exported))

    print(deployment_text)
    print(f"SUMMARY: {summary_path}")
    print(f"GRAPH_JSON: {graph_json}")
    print(f"ANNOTATION: {annotation_path}")
    print(f"ANNOTATION_JSON: {annotation_json}")
    print(f"LAYOUT_ANALYSIS: {layout_path}")
    print(f"LAYOUT_JSON: {layout_json}")
    print(f"REGION_PLAN: {region_path}")
    print(f"REGION_JSON: {region_json}")
    print(f"DEPLOYMENT_GRAPH: {deployment_path}")
    print(f"DEPLOYMENT_JSON: {deployment_json}")
    if export_path:
        print(f"EXPORTED_SUBMODELS: {export_path}")


def cmd_score_onnx_dir(args: argparse.Namespace) -> None:
    root = Path(args.onnx_root)
    paths = sorted(root.glob(args.glob))
    if args.limit_input and args.limit_input > 0:
        paths = paths[: args.limit_input]
    rules = RuleDB.load(Path(args.rules))
    items = score_onnx_portfolio(paths, rules)
    text = render_portfolio(items, limit=args.limit)
    safe_glob = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.glob).strip("_") or "all"
    safe_name = f"{root.name.replace('/', '_') or 'onnx'}_{safe_glob}"
    report = write_report(f"onnx_portfolio_{safe_name}.tsv", text)
    json_path = DEFAULT_REPORT_DIR / f"onnx_portfolio_{safe_name}.json"
    save_portfolio_json(items, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_compile_cmodel(args: argparse.Namespace) -> None:
    result = run_compile_cmodel(
        CompileConfig(
            model=args.model,
            workspace=args.workspace,
            output_root=args.output_root,
            layout=args.layout,
            arch_path=args.arch_path,
            run_cv_model=not args.skip_cv_model,
            seed=args.seed,
            timeout_sec=args.timeout_sec,
            checkpoint=args.checkpoint,
        )
    )
    text = render_compile_result(result)
    path = write_report(f"compile_cmodel_{args.model.replace('/', '.')}.txt", text)
    print(text)
    print(f"REPORT: {path}")


def cmd_pack(args: argparse.Namespace) -> None:
    result = run_pack(
        PackConfig(
            compile_output_dir=args.compile_output_dir,
            packer_output_dir=args.packer_output_dir,
            workspace=args.workspace,
            model_packer_dir=args.model_packer_dir,
            force=not args.no_force,
            timeout_sec=args.timeout_sec,
            rram_only=args.rram_only,
        )
    )
    text = "\n".join(
        [
            f"packer_output_dir: {result.packer_output_dir}",
            f"returncode: {result.returncode}",
            f"has_config: {result.has_config}",
            f"has_op_insts: {result.has_op_insts}",
            f"has_params_wei: {result.has_params_wei}",
            f"has_params_ch: {result.has_params_ch}",
        ]
    )
    path = write_report("pack_result.txt", text)
    print(text)
    print(f"REPORT: {path}")


def cmd_board_run(args: argparse.Namespace) -> None:
    result = run_board(
        BoardConfig(
            packer_dir=args.packer_dir,
            board=args.board,
            password=args.password,
            board_work_dir=args.board_work_dir,
            runner=args.runner,
            model_name=args.model_name,
            local_runner=args.local_runner,
            runner_args=args.runner_args,
            log_path=args.log_path,
            timeout_sec=args.timeout_sec,
        )
    )
    text = render_board_result(result)
    path = write_report("board_result.txt", text)
    print(text)
    print(f"REPORT: {path}")


def cmd_deploy_loop(args: argparse.Namespace) -> None:
    result = run_deploy_loop(
        DeployLoopConfig(
            model=args.model,
            workspace=args.workspace,
            work_root=args.work_root,
            layout=args.layout,
            arch_path=args.arch_path,
            run_cv_model=not args.skip_cv_model,
            run_board=args.run_board,
            board=args.board,
            board_password=args.password,
            board_work_dir=args.board_work_dir,
            local_runner=args.local_runner,
            runner_args=" ".join(args.runner_args),
            checkpoint=args.checkpoint,
        )
    )
    text = render_deploy_loop(result)
    path = write_report(f"deploy_loop_{args.model.replace('/', '.')}.txt", text)
    print(text)
    print(f"REPORT: {path}")


def cmd_suggest_rule(args: argparse.Namespace) -> None:
    rule = suggest_rule_from_result(Path(args.result_json))
    text = render_rule_update(rule)
    path = write_report("suggested_rule_update.json", text)
    print(text)
    print(f"REPORT: {path}")


def cmd_localize_failure(args: argparse.Namespace) -> None:
    localization = localize_failure(
        Path(args.result_json),
        Path(args.region_plan_json) if args.region_plan_json else None,
    )
    text = render_failure_localization(localization)
    stem = Path(args.result_json).stem
    report = write_report(f"failure_localization_{stem}.md", text)
    json_path = DEFAULT_REPORT_DIR / f"failure_localization_{stem}.json"
    save_failure_localization(localization, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_plan_retry(args: argparse.Namespace) -> None:
    plan = plan_retries(Path(args.localization_json))
    text = render_retry_plan(plan)
    stem = Path(args.localization_json).stem
    report = write_report(f"retry_plan_{stem}.md", text)
    json_path = DEFAULT_REPORT_DIR / f"retry_plan_{stem}.json"
    save_retry_plan(plan, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_profile_source(args: argparse.Namespace) -> None:
    profile = profile_source_tree(Path(args.source_root), limit=args.limit)
    text = render_source_profile(profile)
    safe_name = Path(args.source_root).name.replace("/", "_")
    report = write_report(f"source_profile_{safe_name}.md", text)
    json_path = DEFAULT_REPORT_DIR / f"source_profile_{safe_name}.json"
    save_source_profile(profile, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_new_case(args: argparse.Namespace) -> None:
    shape = [int(x) for x in args.input_shape.split(",") if x.strip()]
    case = generate_case(
        case_name=args.case_name,
        model_family=args.model_family,
        source_root=Path(args.source_root),
        input_shape=shape,
        onnx_glob=args.onnx_glob,
        checkpoint=args.checkpoint,
        representative_data=args.representative_data,
    )
    text = render_generated_case(case)
    out_path = Path(args.output) if args.output else DEFAULT_REPORT_DIR.parent / "examples" / f"{args.case_name}.json"
    save_generated_case(case, out_path)
    print(text)
    print(f"CASE: {out_path}")


def cmd_calibrate_costs(args: argparse.Namespace) -> None:
    calibration = calibrate_costs(Path(args.result_root))
    text = render_cost_calibration(calibration)
    safe_name = Path(args.result_root).name.replace("/", "_")
    report = write_report(f"cost_calibration_{safe_name}.md", text)
    json_path = DEFAULT_REPORT_DIR / f"cost_calibration_{safe_name}.json"
    save_cost_calibration(calibration, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_validate_models(args: argparse.Namespace) -> None:
    models = []
    if args.models:
        models.extend(item.strip() for item in args.models.split(",") if item.strip())
    if args.models_file:
        models.extend(
            line.strip()
            for line in Path(args.models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    report_obj = validate_models(
        models=models,
        layout=args.layout,
        workspace=args.workspace,
        work_root=args.work_root,
        checkpoint=args.checkpoint,
        run_board=args.run_board,
        skip_cv_model=args.skip_cv_model,
        board=args.board,
        password=args.password,
        board_work_dir=args.board_work_dir,
    )
    text = render_validation_report(report_obj)
    report = write_report(f"model_validation_{args.name}.tsv", text)
    json_path = DEFAULT_REPORT_DIR / f"model_validation_{args.name}.json"
    save_validation_report(report_obj, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_execute_retry(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_REPORT_DIR.parent / "work" / "retry_exports" / Path(args.region_plan_json).stem
    execution = execute_retry_plan(Path(args.retry_plan_json), Path(args.region_plan_json), output_dir)
    text = render_retry_execution(execution)
    stem = Path(args.retry_plan_json).stem
    report = write_report(f"retry_execution_{stem}.md", text)
    json_path = DEFAULT_REPORT_DIR / f"retry_execution_{stem}.json"
    save_retry_execution(execution, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_deep_search(args: argparse.Namespace) -> None:
    rules = RuleDB.load(Path(args.rules))
    result = run_deep_search(
        onnx_path=Path(args.onnx),
        rules=rules,
        report_dir=DEFAULT_REPORT_DIR,
        effective_budget_bytes=int(args.effective_budget_bytes) if args.effective_budget_bytes else 0,
        policy_path=Path(args.policy) if args.policy else DEFAULT_DEPLOYMENT_POLICY,
    )
    text = render_deep_search_result(result)
    stem = Path(args.onnx).stem
    report = write_report(f"deep_search_{stem}.tsv", text)
    json_path = DEFAULT_REPORT_DIR / f"deep_search_{stem}.json"
    save_deep_search_result(result, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_generate_package(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not args.output_dir:
        stem = Path(args.deep_search_json).stem
        output_dir = DEFAULT_REPORT_DIR.parent / "work" / "deployment_packages" / stem
    package = generate_deployment_package(
        deep_search_json=Path(args.deep_search_json),
        output_dir=output_dir,
        candidate_name=args.candidate,
    )
    text = render_deployment_package(package)
    report = write_report(f"deployment_package_{Path(output_dir).name}.md", text)
    print(text)
    print(f"REPORT: {report}")


def cmd_generate_contract(args: argparse.Namespace) -> None:
    package_dir = Path(args.package_dir)
    contract = build_package_contract(package_dir)
    json_path = package_dir / "package_contract.json"
    md_path = package_dir / "package_contract.md"
    save_package_contract(contract, json_path, md_path)
    text = render_package_contract(contract)
    report = write_report(f"package_contract_{package_dir.name}.md", text)
    print(text)
    print(f"CONTRACT_JSON: {json_path}")
    print(f"CONTRACT_MD: {md_path}")
    print(f"REPORT: {report}")


def cmd_validate_package(args: argparse.Namespace) -> None:
    result = validate_package(
        package_dir=Path(args.package_dir),
        workspace=Path(args.workspace),
        work_root=args.work_root,
        run_board_flag=args.run_board,
        board=args.board,
        password=args.password,
        board_work_dir=args.board_work_dir,
    )
    text = render_package_validation(result)
    safe_name = Path(args.package_dir).name
    report = write_report(f"package_validation_{safe_name}.tsv", text)
    json_path = DEFAULT_REPORT_DIR / f"package_validation_{safe_name}.json"
    save_package_validation(result, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_health_report(args: argparse.Namespace) -> None:
    roots = [Path(x) for x in args.roots]
    report_obj = build_health_report(roots)
    text = render_health_report(report_obj, limit=args.limit)
    report = write_report(args.output_md, text)
    json_path = DEFAULT_REPORT_DIR / args.output_json
    save_health_report_json(report_obj, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_rewrite_backlog(args: argparse.Namespace) -> None:
    backlog = build_rewrite_backlog(Path(args.health_json))
    text = render_rewrite_backlog(backlog, limit=args.limit)
    report = write_report(args.output_md, text)
    json_path = DEFAULT_REPORT_DIR / args.output_json
    save_rewrite_backlog_json(backlog, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_analyze_outliers(args: argparse.Namespace) -> None:
    if args.boundary_csv:
        metrics = load_boundary_csv(Path(args.boundary_csv))
        safe_name = Path(args.boundary_csv).stem
    else:
        keys = [item.strip() for item in args.keys.split(",") if item.strip()] if args.keys else None
        metrics = analyze_npz_pairs(Path(args.ref_npz), Path(args.board_npz), keys=keys)
        safe_name = f"{Path(args.board_npz).stem}_vs_{Path(args.ref_npz).stem}"
    text = render_outlier_report(metrics, limit=args.limit)
    report = write_report(f"outlier_analysis_{safe_name}.tsv", text)
    json_path = DEFAULT_REPORT_DIR / f"outlier_analysis_{safe_name}.json"
    save_outlier_json(metrics, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_quant_diagnostics(args: argparse.Namespace) -> None:
    keys = [item.strip() for item in args.keys.split(",") if item.strip()] if args.keys else None
    channel_axis = int(args.channel_axis) if args.channel_axis != "" else None
    report_obj = analyze_npz(
        Path(args.npz),
        keys=keys,
        channel_axis=channel_axis,
        max_channel_items=args.max_channel_items,
    )
    text = render_quant_diagnostics(report_obj, limit=args.limit)
    safe_name = Path(args.npz).stem.replace("/", "_")
    report = write_report(f"quant_diagnostics_{safe_name}.tsv", text)
    json_path = Path(args.output_json) if args.output_json else DEFAULT_REPORT_DIR / f"quant_diagnostics_{safe_name}.json"
    save_quant_diagnostics(report_obj, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_production_plan(args: argparse.Namespace) -> None:
    plan = build_production_plan(
        model_name=args.model,
        case_path=Path(args.case) if args.case else None,
        allow_approx=not args.no_approx,
        latency_first=not args.accuracy_first,
    )
    text = render_production_plan(plan)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model).strip("_") or "model"
    report = write_report(f"production_plan_{safe_name}.md", text)
    json_path = Path(args.output_json) if args.output_json else DEFAULT_REPORT_DIR / f"production_plan_{safe_name}.json"
    save_production_plan(plan, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_remote_train(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    plan = build_remote_training_plan(config_path, args.profile, DEFAULT_REPORT_DIR)
    if args.action == "plan":
        text = render_remote_training_plan(plan)
        report = write_report(f"remote_training_plan_{args.profile}.md", text)
        json_path = DEFAULT_REPORT_DIR / f"remote_training_plan_{args.profile}.json"
        save_remote_training_plan(plan, json_path)
        print(text)
        print(f"REPORT: {report}")
        print(f"JSON: {json_path}")
        return

    if args.action == "submit":
        result = submit_remote_training(plan, timeout=args.timeout_sec)
    else:
        profile = load_remote_training_profile(config_path, args.profile)
        if args.action == "status":
            result = status_remote_training(profile, timeout=args.timeout_sec)
        elif args.action == "fetch":
            result = fetch_remote_training(profile, timeout=args.timeout_sec)
        elif args.action == "stop":
            result = stop_remote_training(profile, timeout=args.timeout_sec)
        else:
            raise ValueError(f"unknown remote-train action: {args.action}")

    text = "\n".join(
        [
            f"action: {result.action}",
            f"profile: {result.profile}",
            f"ok: {result.ok}",
            f"message: {result.message}",
            f"generated_script: {result.generated_script}",
            "",
            result.log,
        ]
    )
    report = write_report(f"remote_training_{args.action}_{args.profile}.log", text)
    print(text)
    print(f"REPORT: {report}")


def cmd_artifact_manifest(args: argparse.Namespace) -> None:
    paths = [Path(item) for item in args.paths]
    manifest = build_artifact_manifest(args.name, paths)
    text = render_artifact_manifest(manifest, limit=args.limit)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.name).strip("_") or "artifacts"
    report = write_report(f"artifact_manifest_{safe_name}.tsv", text)
    json_path = Path(args.output_json) if args.output_json else DEFAULT_REPORT_DIR / f"artifact_manifest_{safe_name}.json"
    save_artifact_manifest(manifest, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def cmd_export_conv_splits(args: argparse.Namespace) -> None:
    source = Path(args.onnx)
    output_dir = Path(args.output_dir)
    if args.mode == "input-output":
        exports = export_conv_input_output_channel_splits(
            source,
            node_index=args.node_index,
            output_dir=output_dir,
            output_chunk_channels=args.output_chunk_channels,
            input_chunk_channels=args.input_chunk_channels,
        )
    elif args.mode == "output":
        exports = export_conv_output_channel_splits(
            source,
            node_index=args.node_index,
            output_dir=output_dir,
            chunk_channels=args.output_chunk_channels,
            include_following_relu=args.include_following_relu,
        )
    elif args.mode == "im2col-1x1":
        exports = export_stride_conv_im2col_1x1_splits(
            source,
            node_index=args.node_index,
            output_dir=output_dir,
            output_chunk_channels=args.output_chunk_channels,
            flat_input_chunk_channels=args.input_chunk_channels,
        )
    else:
        raise ValueError(f"unknown mode: {args.mode}")
    lines = [
        "split_id\tonnx_path\tinput_name\toutput_name\toc_start\toc_end\tic_start\tic_end",
    ]
    manifest_lines = ["region_id\tonnx_path\tstatus\toutputs"]
    for item in exports:
        lines.append(
            f"{item.split_id}\t{item.onnx_path}\t{item.input_name}\t{item.output_name}\t"
            f"{item.channel_start}\t{item.channel_end}\t{item.input_channel_start}\t{item.input_channel_end}"
        )
        manifest_lines.append(f"{item.split_id}\t{item.onnx_path}\texported\t{item.output_name}")
    manifest_path = output_dir / "rhb_submodels.tsv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    text = "\n".join(lines)
    safe_name = source.stem.replace("/", "_")
    report = write_report(f"conv_splits_{safe_name}.tsv", text)
    print(text)
    print(f"PACKAGE: {output_dir}")
    print(f"MANIFEST: {manifest_path}")
    print(f"REPORT: {report}")


def cmd_make_split_contract(args: argparse.Namespace) -> None:
    contract = build_conv_split_contract(
        name=args.name,
        source=args.source,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        height=args.height,
        width=args.width,
        kernel=args.kernel,
        stride=args.stride,
        padding=args.padding,
        input_chunk=args.input_chunk_channels,
        output_chunk=args.output_chunk_channels,
        post=args.post,
        pad_output_to=args.pad_output_to,
    )
    text = render_split_contract(contract)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.name).strip("_") or "conv_split"
    json_path = DEFAULT_REPORT_DIR / f"split_contract_{safe_name}.json"
    md_path = DEFAULT_REPORT_DIR / f"split_contract_{safe_name}.md"
    save_split_contract(contract, json_path, md_path)
    print(text)
    print(f"JSON: {json_path}")
    print(f"REPORT: {md_path}")


def cmd_execute_backlog(args: argparse.Namespace) -> None:
    result = execute_backlog(
        backlog_json=Path(args.backlog_json),
        category=args.category,
        output_root=Path(args.output_root),
        validate=args.validate,
        workspace=Path(args.workspace),
        work_root=args.work_root,
        run_board=args.run_board,
        limit=args.limit,
    )
    text = render_backlog_execution(result)
    safe_category = args.category.replace("/", "_")
    report = write_report(f"backlog_execution_{safe_category}.tsv", text)
    json_path = DEFAULT_REPORT_DIR / f"backlog_execution_{safe_category}.json"
    save_backlog_execution(result, json_path)
    print(text)
    print(f"REPORT: {report}")
    print(f"JSON: {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RHB black-box auto configuration framework")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rules = sub.add_parser("summarize-rules", help="Summarize the seed RHB rule database")
    rules.add_argument("--rules", default=str(DEFAULT_RULE_DB))
    rules.set_defaults(func=cmd_summarize_rules)

    scan = sub.add_parser("scan-models", help="Scan model experiment files and classify them")
    scan.add_argument("--models-root", required=True)
    scan.add_argument("--limit", type=int, default=80)
    scan.set_defaults(func=cmd_scan_models)

    plan = sub.add_parser("plan", help="Render an initial RHB/Host deployment plan for a case spec")
    plan.add_argument("--case", required=True)
    plan.add_argument("--rules", default=str(DEFAULT_RULE_DB))
    plan.set_defaults(func=cmd_plan)

    onnx_cmd = sub.add_parser("import-onnx", help="Import an ONNX graph and write a graph summary")
    onnx_cmd.add_argument("--onnx", required=True)
    onnx_cmd.add_argument("--node-limit", type=int, default=120)
    onnx_cmd.set_defaults(func=cmd_import_onnx)

    annotate = sub.add_parser("annotate-onnx", help="Annotate ONNX nodes with initial RHB/Host decisions")
    annotate.add_argument("--onnx", required=True)
    annotate.add_argument("--rules", default=str(DEFAULT_RULE_DB))
    annotate.set_defaults(func=cmd_annotate_onnx)

    layout = sub.add_parser("analyze-layout", help="ACSim-style tile/padding/layout risk analysis for an ONNX graph")
    layout.add_argument("--onnx", required=True)
    layout.add_argument("--data-width-bits", type=int, default=8)
    layout.add_argument("--limit", type=int, default=160)
    layout.set_defaults(func=cmd_analyze_layout)

    optimize = sub.add_parser(
        "optimize-onnx",
        help="Import, annotate, region-partition, and emit a Host/RHB deployment graph",
    )
    optimize.add_argument("--onnx", required=True)
    optimize.add_argument("--rules", default=str(DEFAULT_RULE_DB))
    optimize.add_argument("--effective-budget-bytes", default="")
    optimize.add_argument("--allow-approx-rewrites", action="store_true")
    optimize.add_argument("--export-submodels", action="store_true")
    optimize.add_argument("--export-dir", default="")
    optimize.add_argument("--split-multi-output", action="store_true")
    optimize.add_argument("--node-limit", type=int, default=120)
    optimize.add_argument("--data-width-bits", type=int, default=8)
    optimize.set_defaults(func=cmd_optimize_onnx)

    portfolio = sub.add_parser("score-onnx-dir", help="Score many ONNX files as candidate RHB/Host subgraphs")
    portfolio.add_argument("--onnx-root", required=True)
    portfolio.add_argument("--glob", default="*.onnx")
    portfolio.add_argument("--rules", default=str(DEFAULT_RULE_DB))
    portfolio.add_argument("--limit", type=int, default=80)
    portfolio.add_argument("--limit-input", type=int, default=0)
    portfolio.set_defaults(func=cmd_score_onnx_dir)

    cc = sub.add_parser("compile-cmodel", help="Run make cv_model/compile/cmodel for one model")
    cc.add_argument("--model", required=True)
    cc.add_argument("--workspace", default="/root/demo")
    cc.add_argument("--output-root", default="artifacts/rhb_auto_config_framework/work/compile")
    cc.add_argument("--layout", default="input0=BWC")
    cc.add_argument("--arch-path", default="arch_16.yaml,arch_256.yaml")
    cc.add_argument("--seed", type=int, default=1)
    cc.add_argument("--skip-cv-model", action="store_true")
    cc.add_argument("--timeout-sec", type=int, default=1800)
    cc.add_argument("--checkpoint", default="", help="Optional checkpoint path exported as COMPLETIONFORMER_HW_CKPT")
    cc.set_defaults(func=cmd_compile_cmodel)

    pack = sub.add_parser("pack", help="Run Model-Packer on a compiled model directory")
    pack.add_argument("--compile-output-dir", required=True)
    pack.add_argument("--packer-output-dir", required=True)
    pack.add_argument("--workspace", default="/root/demo")
    pack.add_argument("--model-packer-dir", default="Model-Packer")
    pack.add_argument("--no-force", action="store_true")
    pack.add_argument("--rram-only", action="store_true", help="Keep generated packer config as rram_only=true")
    pack.add_argument("--timeout-sec", type=int, default=1200)
    pack.set_defaults(func=cmd_pack)

    board = sub.add_parser("board-run", help="Transfer a packer to board and run deploy.py or a custom runner")
    board.add_argument("--packer-dir", required=True)
    board.add_argument("--board", default="root@192.168.115.122")
    board.add_argument("--password", default="root")
    board.add_argument("--board-work-dir", default="/home/root/workspace/demo_vp_xj/packers/rhb_auto_probe")
    board.add_argument("--runner", default="/home/root/workspace/demo_vp_xj/deploy.py")
    board.add_argument("--model-name", default="")
    board.add_argument("--local-runner", default="")
    board.add_argument("--runner-args", nargs="*", default=[])
    board.add_argument("--log-path", default="")
    board.add_argument("--timeout-sec", type=int, default=1800)
    board.set_defaults(func=cmd_board_run)

    loop = sub.add_parser("deploy-loop", help="Run compile, cmodel, pack, and optionally board in one command")
    loop.add_argument("--model", required=True)
    loop.add_argument("--workspace", default="/root/demo")
    loop.add_argument("--work-root", default="artifacts/rhb_auto_config_framework/work")
    loop.add_argument("--layout", default="input0=BWC")
    loop.add_argument("--arch-path", default="arch_16.yaml,arch_256.yaml")
    loop.add_argument("--skip-cv-model", action="store_true")
    loop.add_argument("--run-board", action="store_true")
    loop.add_argument("--board", default="root@192.168.115.122")
    loop.add_argument("--password", default="root")
    loop.add_argument("--board-work-dir", default="/home/root/workspace/demo_vp_xj/packers/rhb_auto_probe")
    loop.add_argument("--local-runner", default="")
    loop.add_argument("--runner-args", nargs="*", default=[])
    loop.add_argument("--checkpoint", default="", help="Optional checkpoint path exported as COMPLETIONFORMER_HW_CKPT")
    loop.set_defaults(func=cmd_deploy_loop)

    feedback = sub.add_parser("suggest-rule", help="Generate a reviewed rule-update draft from a result JSON")
    feedback.add_argument("--result-json", required=True)
    feedback.set_defaults(func=cmd_suggest_rule)

    localize = sub.add_parser("localize-failure", help="Classify a compile/board failure and suggest likely causes")
    localize.add_argument("--result-json", required=True)
    localize.add_argument("--region-plan-json", default="")
    localize.set_defaults(func=cmd_localize_failure)

    retry = sub.add_parser("plan-retry", help="Generate retry actions from a failure localization JSON")
    retry.add_argument("--localization-json", required=True)
    retry.set_defaults(func=cmd_plan_retry)

    profile = sub.add_parser("profile-source", help="Profile a PyTorch source tree for RHB-risky ops")
    profile.add_argument("--source-root", required=True)
    profile.add_argument("--limit", type=int, default=300)
    profile.set_defaults(func=cmd_profile_source)

    case = sub.add_parser("new-case", help="Generate a new model case spec")
    case.add_argument("--case-name", required=True)
    case.add_argument("--model-family", required=True)
    case.add_argument("--source-root", required=True)
    case.add_argument("--input-shape", required=True, help="Comma-separated, e.g. 1,3,128,128")
    case.add_argument("--onnx-glob", default="")
    case.add_argument("--checkpoint", default="")
    case.add_argument("--representative-data", default="")
    case.add_argument("--output", default="")
    case.set_defaults(func=cmd_new_case)

    calibrate = sub.add_parser("calibrate-costs", help="Collect measured board latency/counters from result JSON files")
    calibrate.add_argument("--result-root", required=True)
    calibrate.set_defaults(func=cmd_calibrate_costs)

    validate = sub.add_parser("validate-models", help="Batch deploy-loop validation for model names")
    validate.add_argument("--models", default="", help="Comma-separated model names")
    validate.add_argument("--models-file", default="")
    validate.add_argument("--name", default="batch")
    validate.add_argument("--workspace", default="/root/demo")
    validate.add_argument("--work-root", default="artifacts/rhb_auto_config_framework/work")
    validate.add_argument("--layout", default="input0=BWC")
    validate.add_argument("--checkpoint", default="")
    validate.add_argument("--skip-cv-model", action="store_true")
    validate.add_argument("--run-board", action="store_true")
    validate.add_argument("--board", default="root@192.168.115.122")
    validate.add_argument("--password", default="root")
    validate.add_argument("--board-work-dir", default="/home/root/workspace/demo_vp_xj/packers/rhb_auto_probe")
    validate.set_defaults(func=cmd_validate_models)

    execute_retry = sub.add_parser("execute-retry", help="Execute supported retry-plan actions")
    execute_retry.add_argument("--retry-plan-json", required=True)
    execute_retry.add_argument("--region-plan-json", required=True)
    execute_retry.add_argument("--output-dir", default="")
    execute_retry.set_defaults(func=cmd_execute_retry)

    deep = sub.add_parser("deep-search", help="Enumerate Host/RHB deployment and rewrite strategies for one ONNX graph")
    deep.add_argument("--onnx", required=True)
    deep.add_argument("--rules", default=str(DEFAULT_RULE_DB))
    deep.add_argument("--policy", default=str(DEFAULT_DEPLOYMENT_POLICY))
    deep.add_argument("--effective-budget-bytes", default="")
    deep.set_defaults(func=cmd_deep_search)

    package = sub.add_parser("generate-package", help="Generate a deployment package from a deep-search result")
    package.add_argument("--deep-search-json", required=True)
    package.add_argument("--candidate", default="", help="Candidate name; defaults to best_candidate")
    package.add_argument("--output-dir", default="")
    package.set_defaults(func=cmd_generate_package)

    contract = sub.add_parser("generate-contract", help="Generate an explicit ACSim-style package contract for an existing package")
    contract.add_argument("--package-dir", required=True)
    contract.set_defaults(func=cmd_generate_contract)

    validate_package_cmd = sub.add_parser("validate-package", help="Compile/cmodel/pack/board validate generated RHB ONNX in a deployment package")
    validate_package_cmd.add_argument("--package-dir", required=True)
    validate_package_cmd.add_argument("--workspace", default="/root/demo")
    validate_package_cmd.add_argument("--work-root", default="artifacts/rhb_auto_config_framework/work")
    validate_package_cmd.add_argument("--run-board", action="store_true")
    validate_package_cmd.add_argument("--board", default="root@192.168.115.122")
    validate_package_cmd.add_argument("--password", default="root")
    validate_package_cmd.add_argument("--board-work-dir", default="/home/root/workspace/demo_vp_xj/packers/rhb_auto_probe")
    validate_package_cmd.set_defaults(func=cmd_validate_package)

    health = sub.add_parser("health-report", help="Summarize compile/cmodel/pack/board health from result JSON files")
    health.add_argument(
        "--roots",
        nargs="+",
        default=[
            str(DEFAULT_REPORT_DIR),
            str(DEFAULT_REPORT_DIR.parent / "work" / "reports"),
        ],
        help="Report directories or JSON files to scan",
    )
    health.add_argument("--limit", type=int, default=120)
    health.add_argument("--output-md", default="framework_health_report.md")
    health.add_argument("--output-json", default="framework_health_report.json")
    health.set_defaults(func=cmd_health_report)

    backlog = sub.add_parser("rewrite-backlog", help="Generate prioritized rewrite/retry actions from a health report")
    backlog.add_argument("--health-json", default=str(DEFAULT_REPORT_DIR / "framework_health_report.json"))
    backlog.add_argument("--limit", type=int, default=120)
    backlog.add_argument("--output-md", default="framework_rewrite_backlog.md")
    backlog.add_argument("--output-json", default="framework_rewrite_backlog.json")
    backlog.set_defaults(func=cmd_rewrite_backlog)

    outliers = sub.add_parser(
        "analyze-outliers",
        help="Analyze activation outliers, range compression, and int8 saturation from ref/board tensors",
    )
    outliers.add_argument("--ref-npz", default="", help="NPZ containing reference tensors")
    outliers.add_argument("--board-npz", default="", help="NPZ containing board tensors")
    outliers.add_argument("--keys", default="", help="Optional comma-separated tensor keys")
    outliers.add_argument(
        "--boundary-csv",
        default="",
        help="CSV with mode,key,l1,rmse,corr,board_min,board_max,ref_min,ref_max columns",
    )
    outliers.add_argument("--limit", type=int, default=120)
    outliers.set_defaults(func=cmd_analyze_outliers)

    quant_diag = sub.add_parser(
        "quant-diagnostics",
        help="Software-side int8 quantization, outlier, kurtosis, and saturation diagnostics for feature NPZs",
    )
    quant_diag.add_argument("--npz", required=True, help="NPZ containing calibration or boundary feature tensors")
    quant_diag.add_argument("--keys", default="", help="Optional comma-separated tensor keys")
    quant_diag.add_argument(
        "--channel-axis",
        default="",
        help="Optional channel axis for per-channel diagnostics, e.g. 1 for BCHW",
    )
    quant_diag.add_argument("--max-channel-items", type=int, default=512)
    quant_diag.add_argument("--limit", type=int, default=120)
    quant_diag.add_argument("--output-json", default="")
    quant_diag.set_defaults(func=cmd_quant_diagnostics)

    production = sub.add_parser(
        "production-plan",
        help="Render the production Host/RHB closed-loop plan for a model case",
    )
    production.add_argument("--model", required=True)
    production.add_argument("--case", default="")
    production.add_argument("--no-approx", action="store_true", help="Disable approximate rewrite/retraining path")
    production.add_argument("--accuracy-first", action="store_true", help="Prefer accuracy over launch-count minimization")
    production.add_argument("--output-json", default="")
    production.set_defaults(func=cmd_production_plan)

    remote_train = sub.add_parser(
        "remote-train",
        help="Template, submit, monitor, fetch, or stop SSH-based hardware-aligned training jobs",
    )
    remote_train.add_argument("--profile", required=True)
    remote_train.add_argument("--config", default=str(DEFAULT_REMOTE_TRAINING_CONFIG))
    remote_train.add_argument("--action", choices=["plan", "submit", "status", "fetch", "stop"], default="plan")
    remote_train.add_argument("--timeout-sec", type=int, default=300)
    remote_train.set_defaults(func=cmd_remote_train)

    artifact_manifest = sub.add_parser(
        "artifact-manifest",
        help="Generate SHA256/mtime/size manifest for checkpoints, ONNX exports, or packer directories",
    )
    artifact_manifest.add_argument("--name", required=True)
    artifact_manifest.add_argument("--paths", nargs="+", required=True)
    artifact_manifest.add_argument("--limit", type=int, default=120)
    artifact_manifest.add_argument("--output-json", default="")
    artifact_manifest.set_defaults(func=cmd_artifact_manifest)

    conv_split = sub.add_parser("export-conv-splits", help="Export exact Conv split ONNX submodels")
    conv_split.add_argument("--onnx", required=True)
    conv_split.add_argument("--node-index", type=int, default=0)
    conv_split.add_argument("--output-dir", required=True)
    conv_split.add_argument("--mode", choices=["input-output", "output", "im2col-1x1"], default="input-output")
    conv_split.add_argument("--output-chunk-channels", type=int, default=32)
    conv_split.add_argument("--input-chunk-channels", type=int, default=8)
    conv_split.add_argument("--include-following-relu", action="store_true")
    conv_split.set_defaults(func=cmd_export_conv_splits)

    split_contract = sub.add_parser(
        "make-split-contract",
        help="Generate a Host/RHB exact Conv split contract with optional post-sum affine/ReLU",
    )
    split_contract.add_argument("--name", required=True)
    split_contract.add_argument("--source", default="")
    split_contract.add_argument("--in-channels", type=int, required=True)
    split_contract.add_argument("--out-channels", type=int, required=True)
    split_contract.add_argument("--height", type=int, required=True)
    split_contract.add_argument("--width", type=int, required=True)
    split_contract.add_argument("--kernel", type=int, default=3)
    split_contract.add_argument("--stride", type=int, default=1)
    split_contract.add_argument("--padding", type=int, default=1)
    split_contract.add_argument("--input-chunk-channels", type=int, default=64)
    split_contract.add_argument("--output-chunk-channels", type=int, default=0)
    split_contract.add_argument("--post", choices=["none", "bias", "bias_relu", "bn_relu"], default="none")
    split_contract.add_argument("--pad-output-to", type=int, default=0)
    split_contract.set_defaults(func=cmd_make_split_contract)

    execute_backlog_cmd = sub.add_parser("execute-backlog", help="Execute supported actions from rewrite-backlog")
    execute_backlog_cmd.add_argument("--backlog-json", default=str(DEFAULT_REPORT_DIR / "framework_rewrite_backlog.json"))
    execute_backlog_cmd.add_argument("--category", default="conv_ic_oc_split")
    execute_backlog_cmd.add_argument("--output-root", default="artifacts/rhb_auto_config_framework/work/rewrite_splits_auto")
    execute_backlog_cmd.add_argument("--validate", action="store_true")
    execute_backlog_cmd.add_argument("--run-board", action="store_true")
    execute_backlog_cmd.add_argument("--workspace", default="/root/demo")
    execute_backlog_cmd.add_argument("--work-root", default="artifacts/rhb_auto_config_framework/work")
    execute_backlog_cmd.add_argument("--limit", type=int, default=0)
    execute_backlog_cmd.set_defaults(func=cmd_execute_backlog)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
