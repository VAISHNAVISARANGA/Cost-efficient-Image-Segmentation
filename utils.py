import json, os, platform, random, subprocess, sys
import numpy as np
import torch
import config


def seed_everything(seed=config.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_ces(q, t_cpu_ms, s_disk_bytes):
    """CES per assignment: 0 if T>50 ms, else Q*[1-0.20*T/50-0.10*min(1,S/4MiB)]."""
    if t_cpu_ms > config.TIME_LIMIT_MS:
        return 0.0
    return q * (1 - 0.20 * (t_cpu_ms / config.TIME_LIMIT_MS)
                - 0.10 * min(1.0, s_disk_bytes / config.SIZE_REF_BYTES))


def cpu_model():
    try:
        if sys.platform.startswith("linux"):
            for l in open("/proc/cpuinfo"):
                if l.lower().startswith("model name"):
                    return l.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        if sys.platform.startswith("win"):
            out = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split("\n")
            return [x.strip() for x in out if x.strip()][1]
    except Exception:
        pass
    return platform.processor() or "unknown"


def env_report():
    import cv2
    return {"cpu_model": cpu_model(), "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
            "torch_threads": torch.get_num_threads(),
            "env_threads": {k: os.environ.get(k) for k in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
