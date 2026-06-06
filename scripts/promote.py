"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from src.config import get_settings

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"

# A few metrics worth surfacing in `show`, in display order.
_KEY_METRICS = ("accuracy_overall", "verdict_rate_leaked", "total_cost_usd")

mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
_client = MlflowClient()


def _resolve_version(name: str, config_id: str):
    """Find the registered version whose `config_id` tag equals `config_id`.

    Returns the matching ModelVersion. Applies the multiplicity rules from
    tasks/task2.md: zero matches → error + exit(1); multiple → warn to stdout
    and return the one with the highest MLflow integer version number.
    """
    matches = _client.search_model_versions(
        f"name = '{name}' AND tags.config_id = '{config_id}'"
    )
    if not matches:
        print(f"error: no version found with config_id={config_id}")
        sys.exit(1)
    matches.sort(key=lambda mv: int(mv.version))
    if len(matches) > 1:
        versions = [int(mv.version) for mv in matches]
        latest = versions[-1]
        print(
            f"warning: multiple versions match config_id={config_id} "
            f"(MLflow versions {versions}); using latest ({latest})"
        )
    return matches[-1]


def _current_config_id(name: str, alias: str) -> str:
    """config_id the alias currently points at, or "" if the alias is unset."""
    try:
        mv = _client.get_model_version_by_alias(name=name, alias=alias)
    except RestException:
        return ""
    return mv.tags.get("config_id", "")


def _append_log(alias: str, from_id: str, to_id: str, op: str) -> None:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": alias,
        "from": from_id,
        "to": to_id,
        "op": op,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _last_log_entry(alias: str) -> dict | None:
    """Most recent log entry for `alias`, or None if the log has none."""
    if not LOG_FILE.exists():
        return None
    last = None
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("alias") == alias:
                last = entry
    return last


def cmd_set(args: argparse.Namespace) -> None:
    """args.alias: str, args.config_id: str. See tasks/task2.md → cmd_set."""
    name = args.name
    target = _resolve_version(name, args.config_id)
    current = _current_config_id(name, args.alias)

    _client.set_registered_model_alias(name, args.alias, target.version)
    _append_log(args.alias, current, args.config_id, "set")

    shown_from = current if current else "(unset)"
    print(f"{args.alias}: {shown_from} → {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_show."""
    name = args.name
    try:
        mv = _client.get_model_version_by_alias(name=name, alias=args.alias)
    except RestException:
        print(f"error: alias {args.alias} is not set")
        sys.exit(1)

    tags = dict(mv.tags)
    config_id = tags.pop("config_id", "(none)")
    print(f"{name} @ {args.alias}")
    print(f"  config_id: {config_id}")
    for key in sorted(tags):
        print(f"  {key}: {tags[key]}")

    metrics = _client.get_run(mv.run_id).data.metrics
    for key in _KEY_METRICS:
        if key not in metrics:
            continue
        value = metrics[key]
        if key == "total_cost_usd":
            print(f"  {key}: ${value:.2f}")
        else:
            print(f"  {key}: {value:.2f}")


def cmd_list(args: argparse.Namespace) -> None:
    """No args. See tasks/task2.md → cmd_list."""
    name = args.name
    try:
        model = _client.get_registered_model(name)
    except RestException:
        print("no aliases set")
        return
    aliases = model.aliases  # {alias_name: version_number_str}
    if not aliases:
        print("no aliases set")
        return

    width = max(len(a) for a in aliases)
    for alias in sorted(aliases):
        version = aliases[alias]
        mv = _client.get_model_version(name=name, version=version)
        config_id = mv.tags.get("config_id", "(none)")
        print(f"{alias.ljust(width)} -> {config_id}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_rollback."""
    name = args.name
    try:
        mv = _client.get_model_version_by_alias(name=name, alias=args.alias)
    except RestException:
        print("nothing to roll back")
        return
    current = mv.tags.get("config_id", "")

    entry = _last_log_entry(args.alias)
    if entry is None:
        print(f"no promotion history for alias {args.alias}")
        return
    if entry.get("op") == "rollback":
        print(
            f"error: {args.alias} was just rolled back; "
            "no further history to walk back to"
        )
        return
    if not entry.get("from"):
        print(f"error: {args.alias} has no previous target (first promotion ever)")
        return

    previous = entry["from"]
    target = _resolve_version(name, previous)
    _client.set_registered_model_alias(name, args.alias, target.version)
    _append_log(args.alias, current, previous, "rollback")

    print(f"{args.alias}: {current} → {previous} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
