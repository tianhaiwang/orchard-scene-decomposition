"""Command-line interface for orchard scene decomposition projects."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from .config import STAGES


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--version", action="version", version="Orchard Scene Decomposition 0.8.6"
    )
    commands = argument_parser.add_subparsers(dest="command", required=True)

    def add(name, help_text, project=True):
        subparser = commands.add_parser(name, help=help_text)
        if project:
            subparser.add_argument("--project", required=True)
            subparser.add_argument("--expected-revision")
        subparser.add_argument("--output", help="Save the JSON response to a new file")
        return subparser

    add("schema", "Print all parameter definitions and defaults", False)
    subparser = add("create", "Create an independent project referencing an input PLY")
    subparser.add_argument("--source", required=True)
    add("status", "Inspect the current project state")
    subparser = add("configure", "Apply a complete parameter JSON file")
    subparser.add_argument("--config", required=True)
    subparser = add("run", "Execute one stage using its saved inputs")
    subparser.add_argument("--stage", required=True, choices=STAGES)
    subparser = add("picks", "Set original-row point IDs in endpoint-pair order")
    subparser.add_argument("--stage", required=True, choices=["poles", "wires", "roi"])
    pick_source_group = subparser.add_mutually_exclusive_group(required=True)
    pick_source_group.add_argument("--ids", nargs="*", type=int)
    pick_source_group.add_argument(
        "--file", help="JSON list, or a viewer response containing ids"
    )
    subparser = add("round", "Start or cancel a facility extraction round")
    subparser.add_argument("--stage", required=True, choices=["poles", "wires"])
    subparser.add_argument(
        "--mode", required=True, choices=["append", "replace", "cancel"]
    )
    subparser = add("export", "Export an annotation or original Gaussian subset")
    subparser.add_argument(
        "--name", default="annotated", help="New filename stem inside exports/"
    )
    subparser.add_argument("--labels", nargs="+", type=int)
    subparser.add_argument("--gaussians-only", action="store_true")
    subparser = add("request", "Execute any documented backend JSON request", False)
    subparser.add_argument("--file", required=True)
    return argument_parser


def request_for(args):
    if args.command == "request":
        request = load_json(args.file)
        if not isinstance(request, dict) or not isinstance(request.get("action"), str):
            raise ValueError("Request must be a JSON object with an action string")
        return request
    request = {"action": args.command}
    if hasattr(args, "project"):
        request["project"] = args.project
        if args.expected_revision:
            request["expected_revision"] = args.expected_revision
    if args.command == "create":
        request["source"] = args.source
    elif args.command == "configure":
        request["config"] = load_json(args.config)
    elif args.command == "run":
        request["stage"] = args.stage
    elif args.command == "picks":
        ids = args.ids if args.ids is not None else load_json(args.file)
        if isinstance(ids, dict):
            ids = ids.get("ids")
        if not isinstance(ids, list):
            raise ValueError(
                "Point IDs must be a JSON list or an object containing ids"
            )
        request.update(stage=args.stage, ids=ids)
    elif args.command == "round":
        request.update(
            action=(
                "cancel-facility-round"
                if args.mode == "cancel"
                else "begin-facility-round"
            ),
            stage=args.stage,
        )
        if args.mode != "cancel":
            request["mode"] = args.mode
    elif args.command == "export":
        request.update(name=args.name, annotated=not args.gaussians_only)
        if args.labels is not None:
            request["labels"] = args.labels
    return request


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.output and Path(args.output).exists():
            raise FileExistsError("Choose a new response filename: " + args.output)
        request = request_for(args)
        from .worker import dispatch

        with redirect_stdout(sys.stderr):
            result = dispatch(request)
        payload = {"ok": True, "result": result}
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as response_file:
                response_file.write(text)
        else:
            sys.stdout.write(text)
        return 0
    except Exception as exc:
        json.dump(
            {"ok": False, "error": str(exc), "type": type(exc).__name__},
            sys.stderr,
            ensure_ascii=False,
        )
        sys.stderr.write("\n")
        return 1
