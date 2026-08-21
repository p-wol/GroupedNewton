"""Entry point for Hydra runs, local or through submitit/Slurm.

Design constraints, all of them consequences of debugging jobs on a machine where the
only thing you get back is a Slurm exit code:

1. Everything that identifies the execution context (host, GPU, job id, interpreter,
   library versions, resolved output directory) is written to disk *by the job itself*,
   before any heavy import or allocation. If a job fails, `env.json` tells you whether it
   even reached Python.
2. Any exception is written to `FAILED.txt` in the job's own output directory and then
   re-raised, so that (a) the traceback survives even if the Slurm log is lost, and
   (b) submitit/Hydra still mark the job as failed.
3. `dry_run=true` exercises the entire chain -- composition, submission, module
   environment, imports, CUDA visibility -- without touching the dataset. It is the
   first thing to run on a new cluster, a new module version, or a new config.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def _nvidia_smi() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # nvidia-smi absent or hanging
        return f"<unavailable: {exc!r}>"


def probe_environment(output_dir: Path) -> dict:
    """Collect and persist the execution context. Must not raise."""
    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "cwd": os.getcwd(),
        "output_dir": str(output_dir),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _nvidia_smi(),
    }

    try:
        import torch

        env["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as exc:
        env["torch"] = f"<import failed: {exc!r}>"

    for mod in ("torchvision", "grnewt", "submitit", "hydra"):
        try:
            m = __import__(mod)
            env[mod] = {"file": getattr(m, "__file__", None),
                        "version": getattr(m, "__version__", None)}
        except Exception as exc:
            env[mod] = f"<import failed: {exc!r}>"

    try:
        (output_dir / "env.json").write_text(json.dumps(env, indent=2, default=str))
    except Exception:
        pass

    return env


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = probe_environment(output_dir)
    print(f"[grnewt] host={env['hostname']} job={env['slurm'].get('SLURM_JOB_ID', '-')} "
          f"out={output_dir}", flush=True)

    # Fail immediately, not after 40 minutes of CPU training, if a GPU was requested and
    # is not there. This is the single most common consequence of a wrong --gres.
    if int(cfg.system.device) > -2:
        t = env.get("torch")
        if not isinstance(t, dict) or not t["cuda_available"]:
            node = env["slurm"].get("SLURM_NODELIST", "?")
            allocated = env["slurm"].get("SLURM_JOB_GPUS") or env["slurm"].get("SLURM_STEP_GPUS")
            smi = env["nvidia_smi"]
            if allocated and "No devices were found" in smi:
                raise RuntimeError(
                    f"Slurm allocated GPU(s) {allocated} on node {node}, but the driver "
                    f"enumerates none (nvidia-smi: {smi!r}). This is a node fault, not a "
                    f"configuration error: resubmit with "
                    f"hydra.launcher.exclude={node} and report the node to assist@idris.fr."
                )
            raise RuntimeError(
                f"CUDA requested (system.device={cfg.system.device}) but unavailable on "
                f"{node}.\n  CUDA_VISIBLE_DEVICES = {env['cuda_visible_devices']!r}\n"
                f"  nvidia-smi = {smi}\n"
                f"  LOADEDMODULES = {os.environ.get('LOADEDMODULES', '<unset>')}"
            )

    if cfg.dry_run:
        (output_dir / "DRY_RUN_OK").write_text(json.dumps(env, indent=2, default=str))
        print("[grnewt] dry_run=true: environment probed, exiting before training.",
              flush=True)
        return 0.0

    (output_dir / "STARTED").write_text(env["timestamp"])
    t0 = time.time()
    try:
        from training_hydra import Trainer  # imported late: dry_run must not need it

        trainer = Trainer(cfg, str(output_dir))
        result = trainer.train()
    except BaseException:
        (output_dir / "FAILED.txt").write_text(traceback.format_exc())
        raise
    (output_dir / "DONE").write_text(f"{time.time() - t0:.1f} s")

    # Returned to the launcher/sweeper; LogJobReturnCallback records it.
    return float(result) if isinstance(result, (int, float)) else None


if __name__ == "__main__":
    main()
