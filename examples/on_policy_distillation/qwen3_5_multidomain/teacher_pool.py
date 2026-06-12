#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml


PROCESS_FILE = "teacher_processes.json"
RUNTIME_FILE = "teacher_runtime.json"


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return _expand(config)


def _is_local_host(host: str) -> bool:
    local_names = {"localhost", "127.0.0.1", "0.0.0.0", socket.gethostname(), socket.getfqdn()}
    return host in local_names


def _quote_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _teacher_entries(config: dict, run_dir: Path) -> tuple[list[dict], dict]:
    run = dict(config.get("run") or {})
    teachers = config.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("teachers.yaml must define a non-empty teachers list.")

    entries = []
    runtime_teachers = []
    for teacher in teachers:
        name = teacher.get("name")
        model_path = teacher.get("model_path")
        instances = teacher.get("instances")
        if not name or not isinstance(name, str):
            raise ValueError("Each teacher must define a non-empty name.")
        if not model_path or not isinstance(model_path, str):
            raise ValueError(f"Teacher {name!r} must define model_path.")
        if not isinstance(instances, list) or not instances:
            raise ValueError(f"Teacher {name!r} must define at least one instance.")

        urls = []
        for idx, instance in enumerate(instances):
            host = instance.get("host", "127.0.0.1")
            public_host = instance.get("public_host", host)
            port = int(instance["port"])
            safe_id = f"{name}_{host.replace('.', '-')}_{port}"
            log_path = run_dir / "logs" / f"{safe_id}.log"
            pid_path = run_dir / "pids" / f"{safe_id}.pid"
            url = f"http://{public_host}:{port}/generate"
            urls.append(url)
            entries.append(
                {
                    "teacher": name,
                    "instance_index": idx,
                    "host": host,
                    "public_host": public_host,
                    "port": port,
                    "url": url,
                    "model_path": model_path,
                    "tokenizer_path": teacher.get("tokenizer_path"),
                    "gpus": str(instance.get("gpus", "")),
                    "tp": int(instance.get("tp", 1)),
                    "mem_fraction_static": instance.get("mem_fraction_static"),
                    "extra_args": list(teacher.get("extra_args") or []) + list(instance.get("extra_args") or []),
                    "log_path": str(log_path),
                    "pid_path": str(pid_path),
                }
            )

        runtime_teachers.append(
            {
                "name": name,
                "domains": teacher.get("domains", []),
                "urls": urls,
                "metadata": teacher.get("metadata", {}),
            }
        )

    runtime = {
        "default_teacher": config.get("default_teacher"),
        "teachers": runtime_teachers,
    }
    _validate_runtime_config(runtime)
    return entries, runtime


def _validate_runtime_config(runtime: dict) -> None:
    teachers = runtime.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("Generated teacher_runtime.json must contain a non-empty teachers list.")

    teacher_names = set()
    for teacher in teachers:
        name = teacher.get("name") if isinstance(teacher, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("Generated teacher_runtime.json contains a teacher without a valid name.")
        if name in teacher_names:
            raise ValueError(f"Generated teacher_runtime.json contains duplicate teacher name: {name}")
        teacher_names.add(name)

        domains = teacher.get("domains", [])
        if not isinstance(domains, list) or not all(isinstance(domain, str) for domain in domains):
            raise ValueError(f"Generated teacher_runtime.json teacher {name!r} has invalid domains.")
        metadata = teacher.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Generated teacher_runtime.json teacher {name!r} has invalid metadata.")
        urls = teacher.get("urls")
        if not isinstance(urls, list) or not urls or not all(isinstance(url, str) and url for url in urls):
            raise ValueError(f"Generated teacher_runtime.json teacher {name!r} has invalid urls.")

    default_teacher = runtime.get("default_teacher")
    if len(teachers) > 1 and default_teacher is None:
        raise ValueError("Generated teacher_runtime.json with multiple teachers must define default_teacher.")
    if default_teacher is not None and default_teacher not in teacher_names:
        raise ValueError(f"Generated teacher_runtime.json default_teacher {default_teacher!r} is not defined.")


def _launch_command(entry: dict, run: dict) -> str:
    python = run.get("python", "python3")
    module = run.get("sglang_module", "sglang.launch_server")
    cmd = [
        python,
        "-m",
        module,
        "--model-path",
        entry["model_path"],
        "--host",
        "0.0.0.0",
        "--port",
        str(entry["port"]),
        "--tp",
        str(entry["tp"]),
    ]
    if entry.get("tokenizer_path"):
        cmd += ["--tokenizer-path", entry["tokenizer_path"]]
    if entry.get("mem_fraction_static") is not None:
        cmd += ["--mem-fraction-static", str(entry["mem_fraction_static"])]
    cmd += [str(arg) for arg in entry["extra_args"]]

    env_prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(entry['gpus'])} " if entry["gpus"] else ""
    return (
        f"mkdir -p {shlex.quote(str(Path(entry['log_path']).parent))} "
        f"{shlex.quote(str(Path(entry['pid_path']).parent))}; "
        f"{env_prefix}nohup setsid {_quote_cmd(cmd)} > {shlex.quote(entry['log_path'])} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(entry['pid_path'])}"
    )


def _shell_command(host: str, command: str, ssh_user: str | None) -> list[str]:
    if _is_local_host(host):
        return ["bash", "-lc", command]
    target = f"{ssh_user}@{host}" if ssh_user else host
    return ["ssh", target, command]


def _run_on_host(host: str, command: str, ssh_user: str | None, dry_run: bool) -> None:
    shell_cmd = _shell_command(host, command, ssh_user)
    print("+", _quote_cmd(shell_cmd))
    if not dry_run:
        subprocess.run(shell_cmd, check=True)


def _command_succeeds_on_host(host: str, command: str, ssh_user: str | None) -> bool:
    return (
        subprocess.run(
            _shell_command(host, command, ssh_user),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _pid_alive_command(entry: dict) -> str:
    pid_path = shlex.quote(entry["pid_path"])
    return f"test -f {pid_path} && kill -0 $(cat {pid_path}) 2>/dev/null"


def _verify_processes_alive(entries: list[dict], ssh_user: str | None, dry_run: bool) -> None:
    if dry_run:
        return
    dead = [
        f"{entry['teacher']}@{entry['host']}:{entry['port']} pid={entry['pid_path']}"
        for entry in entries
        if not _command_succeeds_on_host(entry["host"], _pid_alive_command(entry), ssh_user)
    ]
    if dead:
        raise RuntimeError(f"Teacher process exited before runtime config was written: {dead}")


def _wait_for_health(
    entries: list[dict],
    timeout: int,
    interval: int,
    ssh_user: str | None,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    deadline = time.time() + timeout
    pending = {entry["url"].replace("/generate", "/health_generate") for entry in entries}
    while pending and time.time() < deadline:
        _verify_processes_alive(entries, ssh_user, dry_run)
        for health_url in list(pending):
            try:
                with urllib.request.urlopen(health_url, timeout=3) as response:
                    if response.status == 200:
                        print(f"healthy: {health_url}")
                        pending.remove(health_url)
            except Exception:
                pass
        if pending:
            time.sleep(interval)

    if pending:
        raise TimeoutError(f"Timed out waiting for teacher health checks: {sorted(pending)}")


def _wait_for_post_health_stability(
    entries: list[dict],
    seconds: int,
    interval: int,
    ssh_user: str | None,
    dry_run: bool,
) -> None:
    if dry_run or seconds <= 0:
        return
    deadline = time.time() + seconds
    while time.time() < deadline:
        _verify_processes_alive(entries, ssh_user, dry_run)
        time.sleep(min(interval, max(0.0, deadline - time.time())))
    _verify_processes_alive(entries, ssh_user, dry_run)
    _wait_for_health(entries, timeout=max(3, interval), interval=interval, ssh_user=ssh_user, dry_run=dry_run)


def start(args: argparse.Namespace) -> None:
    config = _load_yaml(Path(args.config))
    run = dict(config.get("run") or {})
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "pids").mkdir(exist_ok=True)

    entries, runtime = _teacher_entries(config, run_dir)
    ssh_user = run.get("ssh_user")
    for entry in entries:
        _run_on_host(entry["host"], _launch_command(entry, run), ssh_user, args.dry_run)

    _wait_for_health(
        entries,
        timeout=int(run.get("startup_timeout_seconds", 600)),
        interval=int(run.get("health_interval_seconds", 5)),
        ssh_user=ssh_user,
        dry_run=args.dry_run,
    )
    _wait_for_post_health_stability(
        entries,
        seconds=int(run.get("post_health_stability_seconds", 5)),
        interval=int(run.get("health_interval_seconds", 5)),
        ssh_user=ssh_user,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        (run_dir / PROCESS_FILE).write_text(json.dumps({"ssh_user": ssh_user, "processes": entries}, indent=2) + "\n")
        (run_dir / RUNTIME_FILE).write_text(json.dumps(runtime, indent=2) + "\n")
    print(f"run_dir={run_dir}")
    print(f"runtime_config={run_dir / RUNTIME_FILE}")


def render_runtime(args: argparse.Namespace) -> None:
    config = _load_yaml(Path(args.config))
    run_dir = Path(args.run_dir or "/tmp/slime-opd-teachers").resolve()
    _, runtime = _teacher_entries(config, run_dir)
    output = json.dumps(runtime, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output, end="")


def stop(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    process_path = run_dir / PROCESS_FILE
    if not process_path.exists():
        raise FileNotFoundError(f"Missing process file: {process_path}")
    data = json.loads(process_path.read_text())
    ssh_user = args.ssh_user if args.ssh_user is not None else data.get("ssh_user")
    for entry in data.get("processes", []):
        command = (
            f"if [ -f {shlex.quote(entry['pid_path'])} ]; then "
            f"kill $(cat {shlex.quote(entry['pid_path'])}) 2>/dev/null || true; "
            f"rm -f {shlex.quote(entry['pid_path'])}; "
            "fi"
        )
        _run_on_host(entry["host"], command, ssh_user, args.dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Qwen3.5 OPD SGLang teacher pools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--config", required=True)
    start_parser.add_argument("--run-dir", required=True)
    start_parser.add_argument("--dry-run", action="store_true")
    start_parser.set_defaults(func=start)

    render_parser = subparsers.add_parser("render-runtime")
    render_parser.add_argument("--config", required=True)
    render_parser.add_argument("--run-dir")
    render_parser.add_argument("--output")
    render_parser.set_defaults(func=render_runtime)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--run-dir", required=True)
    stop_parser.add_argument("--ssh-user")
    stop_parser.add_argument("--dry-run", action="store_true")
    stop_parser.set_defaults(func=stop)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
