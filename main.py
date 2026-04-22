"""main.py — entry point for running experiments.

Usage:
    python main.py                          # sim preset, default system
    python main.py --preset real            # real preset
    python main.py --preset headless        # headless preset
    python main.py --system default:real    # override system inside sim preset
    python main.py --estimators lpf:test,kalman:test
    python main.py --set system.workspace.safe_radius=0.05
    python main.py --list                   # show all available presets
"""

import argparse
import sys
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass, MISSING
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin

from src.shared import build_from_registry, plot_logger_chunk, resolve_preset
from src.experiment import EXPERIMENT_REGISTRY, EXPERIMENT_PRESETS
from src.experiment.metrics import Metrics
from src.system import SYSTEM_REGISTRY, SystemParams
from src.experiment.experiment import ExperimentParams


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a control experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--preset", "-p",
        default="sim",
        choices=list(EXPERIMENT_PRESETS),
        metavar="PRESET",
        help=f"experiment preset to run (choices: {', '.join(EXPERIMENT_PRESETS)}). default: sim",
    )

    # Optional field overrides — each maps to a key in ExperimentParams.
    # Only the fields a user would ever want to swap from the CLI are listed.
    overrides = parser.add_argument_group("preset overrides (optional)")
    overrides.add_argument("--system",         metavar="TYPE:PRESET", help="override system")
    overrides.add_argument("--logger",         metavar="TYPE:PRESET", help="override logger")
    overrides.add_argument("--stop-condition", metavar="TYPE:PRESET", help="override stop condition")
    overrides.add_argument(
        "--realtime-visualizer",
        metavar="TYPE:PRESET",
        help="override realtime visualizer (e.g. sim:default, null:default)",
    )
    overrides.add_argument(
        "--offline-visualizer",
        metavar="TYPE:PRESET",
        help="override offline visualizer (e.g. 3d:default, null:default)",
    )
    overrides.add_argument("--progress",       metavar="TYPE:PRESET", help="override progress reporter")
    overrides.add_argument("--plants",         metavar="TYPE:PRESET[,TYPE:PRESET...]", help="override system plants")
    overrides.add_argument("--controllers",    metavar="TYPE:PRESET[,TYPE:PRESET...]", help="override system controllers")
    overrides.add_argument("--estimators",     metavar="TYPE:PRESET[,TYPE:PRESET...]", help="override system estimators")
    overrides.add_argument("--sensor",         metavar="TYPE:PRESET", help="override system sensor")
    overrides.add_argument("--actuator",       metavar="TYPE:PRESET", help="override system actuator")
    overrides.add_argument("--supervisor",     metavar="TYPE:PRESET", help="override system supervisor")
    overrides.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "override any experiment or system field by path, e.g. "
            "--set n_trials=5 or --set system.workspace.safe_radius=0.05"
        ),
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="list available presets and exit",
    )

    parser.add_argument(
        "--graph",
        metavar="PATH_OR_INDEX",
        help=(
            "load a saved logger chunk pickle and plot it. "
            "Pass a path, or pass 1 for the newest log, 2 for the second newest, and so on"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------

def _split_assignment(assignment: str) -> tuple[list[str], str]:
    if "=" not in assignment:
        raise ValueError(f"Expected PATH=VALUE, got {assignment!r}")
    path, value = assignment.split("=", 1)
    path_parts = [part.strip() for part in path.split(".") if part.strip()]
    if not path_parts:
        raise ValueError(f"Expected a non-empty PATH in {assignment!r}")
    return path_parts, value.strip()


def collect_overrides(args: argparse.Namespace) -> dict[str, list[tuple[list[str], str]]]:
    """Return CLI overrides as experiment/system path assignments."""
    overrides = {
        "experiment": [],
        "system": [],
    }

    top_level_mapping = {
        "system": args.system,
        "logger": args.logger,
        "stop_condition": args.stop_condition,   # argparse maps --stop-condition → stop_condition
        "progress": args.progress,
        "realtime_visualizer": args.realtime_visualizer,
        "offline_visualizer": args.offline_visualizer,
    }
    for key, value in top_level_mapping.items():
        if value is not None:
            overrides["experiment"].append(([key], value))

    system_mapping = {
        "plants": args.plants,
        "controllers": args.controllers,
        "estimators": args.estimators,
        "sensor": args.sensor,
        "actuator": args.actuator,
        "supervisor": args.supervisor,
    }
    for key, value in system_mapping.items():
        if value is not None:
            overrides["system"].append(([key], value))

    for assignment in args.set:
        path, value = _split_assignment(assignment)
        bucket = "system" if path[0] == "system" else "experiment"
        if bucket == "system":
            path = path[1:]
            if not path:
                path = ["system"]
        overrides[bucket].append((path, value))

    return overrides


def resolve_spec_string(preset_name: str, overrides: dict) -> str:
    """
    The experiment registry expects a single "type:preset" string.
    We always use the "default" type; the preset name selects the behaviour.

    Overrides are merged directly into the resolved preset dict before build,
    so we return both pieces separately for build_with_overrides().
    """
    return f"default:{preset_name}"


def _field_map(params_cls) -> dict:
    return {f.name: f for f in fields(params_cls)}


def _non_none_union_args(tp):
    origin = get_origin(tp)
    if origin in (UnionType,):
        return [arg for arg in get_args(tp) if arg is not type(None)]
    if str(origin) == "typing.Union":
        return [arg for arg in get_args(tp) if arg is not type(None)]
    return None


def _dataclass_type(tp):
    if tp is None:
        return None
    if isinstance(tp, type) and is_dataclass(tp):
        return tp
    union_args = _non_none_union_args(tp)
    if union_args is None:
        return None
    for arg in union_args:
        if isinstance(arg, type) and is_dataclass(arg):
            return arg
    return None


def _list_item_type(tp):
    origin = get_origin(tp)
    if origin is list:
        args = get_args(tp)
        return args[0] if args else None
    return None


def _field_default(field_info):
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()
    return MISSING


def _path_type(params_cls, path: list[str]):
    current_type = params_cls
    for part in path:
        if not is_dataclass(current_type):
            return None
        field_info = _field_map(current_type).get(part)
        if field_info is None:
            return None
        current_type = field_info.type
        nested_type = _dataclass_type(current_type)
        if nested_type is not None:
            current_type = nested_type
    return current_type


def _path_default(params_cls, path: list[str]):
    current_type = params_cls
    current_value = MISSING
    for idx, part in enumerate(path):
        if not is_dataclass(current_type):
            return MISSING
        field_info = _field_map(current_type).get(part)
        if field_info is None:
            return MISSING
        current_value = _field_default(field_info)
        is_leaf = idx == len(path) - 1
        if is_leaf:
            return current_value
        nested_type = _dataclass_type(field_info.type)
        if nested_type is None or current_value is MISSING:
            return MISSING
        current_type = nested_type
    return current_value


def _get_nested(mapping: dict, path: list[str], default=MISSING):
    cur = mapping
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _set_nested(mapping: dict, path: list[str], value) -> None:
    cur = mapping
    for part in path[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[path[-1]] = value


def _coerce_scalar(raw_value: str, target_type, current_value):
    lowered = raw_value.lower()
    union_args = _non_none_union_args(target_type)
    if union_args is not None:
        if lowered in {"none", "null"}:
            return None
        for arg in union_args:
            try:
                return _coerce_scalar(raw_value, arg, current_value)
            except ValueError:
                continue
        raise ValueError(f"Could not parse {raw_value!r} for {target_type}")

    if target_type is bool or isinstance(current_value, bool):
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Expected a boolean value, got {raw_value!r}")
    if target_type is int or isinstance(current_value, int):
        return int(raw_value)
    if target_type is float or isinstance(current_value, float):
        return float(raw_value)
    if target_type is str or isinstance(current_value, str):
        return raw_value
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value


def _coerce_value(raw_value: str, target_type, current_value):
    if isinstance(current_value, list) or get_origin(target_type) is list:
        item_type = _list_item_type(target_type)
        items = [] if raw_value == "" else [item.strip() for item in raw_value.split(",") if item.strip()]
        return [_coerce_scalar(item, item_type, None) for item in items]
    return _coerce_scalar(raw_value, target_type, current_value)


def _apply_assignments(raw_config: dict, params_cls, assignments: list[tuple[list[str], str]]) -> dict:
    patched = deepcopy(raw_config)
    for path, raw_value in assignments:
        if len(path) > 1 and _get_nested(patched, path[:-1], MISSING) is MISSING:
            parent_default = _path_default(params_cls, path[:-1])
            if parent_default is not MISSING:
                if is_dataclass(parent_default):
                    parent_default = asdict(parent_default)
                elif isinstance(parent_default, dict):
                    parent_default = deepcopy(parent_default)
                _set_nested(patched, path[:-1], parent_default)
        current_value = _get_nested(patched, path, MISSING)
        if current_value is MISSING:
            current_value = _path_default(params_cls, path)
        target_type = _path_type(params_cls, path)
        value = _coerce_value(raw_value, target_type, current_value)
        _set_nested(patched, path, value)
    return patched


def _apply_registry_field_overrides(
    raw_config: dict,
    params_cls,
    spec,
    assignments: list[tuple[list[str], str]],
) -> tuple[dict, list[tuple[object, str]]]:
    registry_fields = spec.registries or {}
    direct_assignments: list[tuple[list[str], str]] = []
    nested_registry_assignments: dict[str, list[tuple[list[str], str]]] = {}

    for path, raw_value in assignments:
        if len(path) > 1 and path[0] in registry_fields:
            nested_registry_assignments.setdefault(path[0], []).append((path[1:], raw_value))
        else:
            direct_assignments.append((path, raw_value))

    patched = _apply_assignments(raw_config, params_cls, direct_assignments)
    cleanup: list[tuple[object, str]] = []

    for field_name, field_assignments in nested_registry_assignments.items():
        current_spec_string = patched.get(field_name)
        if not isinstance(current_spec_string, str) or ":" not in current_spec_string:
            raise ValueError(
                f"Cannot apply nested override to '{field_name}' because it is not a registry spec string."
            )

        sub_registry = registry_fields[field_name]
        sub_type, sub_preset_name = current_spec_string.split(":", 1)
        sub_spec = sub_registry[sub_type]
        sub_raw = _apply_assignments(
            resolve_preset(sub_spec.Presets, sub_preset_name),
            sub_spec.Params,
            field_assignments,
        )
        ephemeral_name = f"__cli_{field_name}__"
        sub_spec.Presets[ephemeral_name] = sub_raw
        cleanup.append((sub_spec, ephemeral_name))
        patched[field_name] = f"{sub_type}:{ephemeral_name}"

    return patched, cleanup


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_experiment(spec_string: str, overrides: dict):
    """
    Build the experiment, applying any CLI overrides on top of the preset.

    If there are no overrides we delegate entirely to build_from_registry.
    If there are overrides we resolve the preset manually, patch it, then build.
    """
    registry = EXPERIMENT_REGISTRY
    type_, preset_name = spec_string.split(":", 1)
    spec = registry[type_]

    experiment_assignments = overrides.get("experiment", [])
    system_assignments = overrides.get("system", [])

    if not experiment_assignments and not system_assignments:
        return build_from_registry(registry, spec_string)

    # Resolve preset (handles "base" inheritance), then patch.
    raw = _apply_assignments(resolve_preset(spec.Presets, preset_name), ExperimentParams, experiment_assignments)

    system_ephemeral_name = None
    system_spec = None
    cleanup_specs: list[tuple[object, str]] = []
    if system_assignments:
        system_spec_string = raw["system"]
        system_type, system_preset_name = system_spec_string.split(":", 1)
        system_spec = SYSTEM_REGISTRY[system_type]
        system_raw, system_cleanup = _apply_registry_field_overrides(
            resolve_preset(system_spec.Presets, system_preset_name),
            SystemParams,
            system_spec,
            system_assignments,
        )
        cleanup_specs.extend(system_cleanup)
        system_ephemeral_name = "__cli_system__"
        system_spec.Presets[system_ephemeral_name] = system_raw
        cleanup_specs.append((system_spec, system_ephemeral_name))
        raw["system"] = f"{system_type}:{system_ephemeral_name}"

    # Re-use build_from_registry's inner loop by passing the patched dict
    # through the same resolution path.  We reconstruct a minimal spec-string
    # that points at an ephemeral preset we inject temporarily.
    _EPHEMERAL = "__cli__"
    spec.Presets[_EPHEMERAL] = raw
    try:
        experiment = build_from_registry(registry, f"{type_}:{_EPHEMERAL}")
    finally:
        spec.Presets.pop(_EPHEMERAL, None)
        for cleanup_spec, cleanup_name in reversed(cleanup_specs):
            cleanup_spec.Presets.pop(cleanup_name, None)

    return experiment


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def list_presets() -> None:
    print("Available experiment presets:\n")
    for name, preset in EXPERIMENT_PRESETS.items():
        base = f"  (extends '{preset['base']}')" if "base" in preset else ""
        print(f"  {name:<12}{base}")
    print()


def _sorted_logger_chunk_paths(log_dir: Path) -> list[Path]:
    return sorted(
        (path for path in log_dir.glob("*.pkl") if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )


def resolve_graph_target(graph_arg: str) -> Path:
    candidate = Path(graph_arg).expanduser()
    if candidate.exists():
        return candidate

    try:
        graph_index = int(graph_arg)
    except ValueError as exc:
        raise FileNotFoundError(
            f"--graph expected an existing path or a positive integer, got {graph_arg!r}"
        ) from exc

    if graph_index <= 0:
        raise ValueError(f"--graph index must be positive, got {graph_index}")

    log_dir = Path("logs")
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    log_paths = sorted(
        (path for path in log_dir.rglob("logger_chunk_*.pkl") if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if not log_paths:
        raise FileNotFoundError(f"No logger chunk pickle files found in {log_dir}")
    if graph_index > len(log_paths):
        raise IndexError(
            f"Requested log #{graph_index}, but only {len(log_paths)} log files were found in {log_dir}"
        )

    return log_paths[graph_index - 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.list:
        list_presets()
        return 0

    if args.graph:
        graph_path = resolve_graph_target(args.graph)
        print(f"[main] plotting log: {graph_path}")
        plot_logger_chunk(graph_path)
        return 0

    spec_string = resolve_spec_string(args.preset, {})
    overrides   = collect_overrides(args)

    print(f"[main] building experiment '{args.preset}'" +
          (f" with overrides: {overrides}" if overrides else ""))

    experiment = build_experiment(spec_string, overrides)
    results    = experiment.run_experiment()

    metrics = Metrics()
    trial_metrics = [metrics.evaluate(r) for r in results]
    summary = metrics.summarize(trial_metrics)
    metrics.print_summary(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
