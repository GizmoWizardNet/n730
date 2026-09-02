#!/usr/bin/env python3
"""
Pick a compatible HuggingFace model, point HF_HOME at the bundled hf-cache
folder, then walk through profiling -> conversion -> interactive inference,
all from one menu.

Run it with:  python setup.py
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT        = Path(__file__).resolve().parent
SRC         = ROOT / "src-code"
CPP_DIR     = SRC / "cpp"
HF_CACHE    = ROOT / "hf-cache"
MODELS_DIR  = ROOT / "models"          # where .n730 + sensitivity_map.json land
STATE_FILE  = ROOT / ".n730_setup_state.json"

PROFILER   = SRC / "profiler.py"
CONVERTER  = SRC / "converter.py"
INFERENCE  = SRC / "inference.py"

IS_WINDOWS = os.name == "nt"
CORE_LIB   = CPP_DIR / ("n730core.dll" if IS_WINDOWS else "n730core.so")
CUDA_LIB   = CPP_DIR / ("n730_cuda.dll" if IS_WINDOWS else "n730_cuda.so")

COMPATIBLE_MODELS = [
    {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "label": "DeepSeek-R1-Distill-Qwen 1.5B",
        "note": "Default / best-tested target. ~3.5GB FP32 download.",
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B Instruct",
        "note": "Smallest, fastest to profile+convert. Good for a first run.",
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B Instruct",
        "note": "Same size class as the DeepSeek distill above.",
    },
    {
        "id": "Qwen/Qwen2-1.5B-Instruct",
        "label": "Qwen2 1.5B Instruct",
        "note": "Older Qwen2 generation, same architecture family.",
    },
    {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "label": "DeepSeek-R1-Distill-Qwen 7B",
        "note": "Much bigger. Long profile/convert time, more VRAM streaming.",
    },
]

class C:
    USE_COLOR = sys.stdout.isatty() and not IS_WINDOWS or os.environ.get("FORCE_COLOR")
    RESET   = "\033[0m"   if USE_COLOR else ""
    BOLD    = "\033[1m"   if USE_COLOR else ""
    DIM     = "\033[2m"   if USE_COLOR else ""
    GREEN   = "\033[32m"  if USE_COLOR else ""
    RED     = "\033[31m"  if USE_COLOR else ""
    YELLOW  = "\033[33m"  if USE_COLOR else ""
    CYAN    = "\033[36m"  if USE_COLOR else ""
    MAGENTA = "\033[35m"  if USE_COLOR else ""


def clear():
    os.system("cls" if IS_WINDOWS else "clear")


def banner():
    print(f"{C.CYAN}{C.BOLD}")
    print(r"   _   _ _____ _____ ___  ")
    print(r"  | \ | |___  |___ // _ \ ")
    print(r"  |  \| |  / /   |_ \ | | |")
    print(r"  | |\  | / /   ___) | |_| |")
    print(r"  |_| \_|/_/   |____/ \___/ ")
    print(f"{C.RESET}{C.DIM}  streamed transformer inference for legacy GPUs{C.RESET}\n")


def rule(char="─", width=60):
    print(C.DIM + char * width + C.RESET)


def info(msg):    print(f"{C.CYAN}  ›{C.RESET} {msg}")
def ok(msg):      print(f"{C.GREEN}  ✓{C.RESET} {msg}")
def warn(msg):    print(f"{C.YELLOW}  ⚠{C.RESET} {msg}")
def err(msg):     print(f"{C.RED}  ✗{C.RESET} {msg}")
def step(n, total, msg): print(f"{C.MAGENTA}  [{n}/{total}]{C.RESET} {C.BOLD}{msg}{C.RESET}")


def prompt_choice(title, options, allow_back=True):
    """options: list of (label, sublabel) tuples. Returns index or None on back."""
    while True:
        print(f"\n{C.BOLD}{title}{C.RESET}")
        rule()
        for i, (label, sub) in enumerate(options, 1):
            print(f"  {C.CYAN}{i}.{C.RESET} {label}")
            if sub:
                print(f"     {C.DIM}{sub}{C.RESET}")
        if allow_back:
            print(f"  {C.CYAN}0.{C.RESET} Back / cancel")
        rule()
        raw = input(f"{C.BOLD}  > {C.RESET}").strip()
        if allow_back and raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        warn("Not a valid choice, try again.")


def prompt_text(msg, default=None):
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{C.BOLD}  {msg}{suffix}: {C.RESET}").strip()
    return raw if raw else default


def prompt_yesno(msg, default=True):
    d = "Y/n" if default else "y/N"
    raw = input(f"{C.BOLD}  {msg} ({d}): {C.RESET}").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def pause():
    input(f"\n{C.DIM}  Press Enter to continue...{C.RESET}")


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "torch": "torch",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
}


def check_python_packages():
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def install_packages(pip_names):
    info(f"Installing: {', '.join(pip_names)}")
    cmd = [sys.executable, "-m", "pip", "install", *pip_names]
    rc = subprocess.call(cmd)
    return rc == 0


def check_cuda_core():
    """Return (core_ok, cuda_ok) for the two native libraries."""
    return CORE_LIB.exists(), CUDA_LIB.exists()


def build_hint():
    if IS_WINDOWS:
        return textwrap.dedent(f"""
        Build the C++ scheduler core:
          cd {CPP_DIR}
          g++ -O3 -march=native -shared -o n730core.dll n730core.cpp

        Build the CUDA kernels (needs CUDA Toolkit 11.4 for the GT 730 / sm_35):
          cd {CPP_DIR}
          nvcc -O3 -arch=sm_35 --shared -lcublas -o n730_cuda.dll n730_cuda.cu
        """).strip()
    return textwrap.dedent(f"""
        Build the C++ scheduler core:
          cd {CPP_DIR}
          g++ -O3 -march=native -shared -fPIC -o n730core.so n730core.cpp

        Build the CUDA kernels (needs CUDA Toolkit 11.4 for the GT 730 / sm_35):
          cd {CPP_DIR}
          nvcc -O3 -arch=sm_35 -shared -Xcompiler -fPIC -lcublas -o n730_cuda.so n730_cuda.cu
        """).strip()


def setup_hf_home():
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE)
    return HF_CACHE

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass

def choose_model(state):
    options = [(m["label"], f"{m['id']} — {m['note']}") for m in COMPATIBLE_MODELS]
    options.append(("Enter a custom HuggingFace repo id", "Must follow the Qwen2-style GQA + SwiGLU + RMSNorm layout"))

    idx = prompt_choice("Select a HuggingFace model", options)
    if idx is None:
        return None
    if idx == len(COMPATIBLE_MODELS):
        repo_id = prompt_text("HuggingFace repo id (e.g. org/model-name)")
        if not repo_id:
            return None
        return repo_id
    model_id = COMPATIBLE_MODELS[idx]["id"]
    state["last_model"] = model_id
    save_state(state)
    return model_id


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


# ─────────────────────────────────────────────────────────────────────────
#  Pipeline stages — each just shells out to the existing scripts so we
#  stay in lockstep with whatever profiler.py / converter.py / inference.py
#  actually do.
# ─────────────────────────────────────────────────────────────────────────

def run(cmd, cwd=None):
    info(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.call(cmd, cwd=cwd or SRC)


def stage_profile(model_id: str) -> Path | None:
    run_dir = MODELS_DIR / slug(model_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_path = run_dir / "sensitivity_map.json"

    step(1, 3, f"Profiling {model_id}")
    rc = run([
        sys.executable, str(PROFILER),
        "--model", model_id,
        "--output", str(sensitivity_path),
    ])
    if rc != 0 or not sensitivity_path.exists():
        err("Profiling failed.")
        return None
    ok(f"Sensitivity map written to {sensitivity_path}")
    return sensitivity_path


def stage_convert(model_id: str, sensitivity_path: Path) -> Path | None:
    run_dir = sensitivity_path.parent
    n730_path = run_dir / "model.n730"

    step(2, 3, f"Converting {model_id} → .n730")
    rc = run([
        sys.executable, str(CONVERTER),
        "--model", model_id,
        "--sensitivity", str(sensitivity_path),
        "--output", str(n730_path),
    ])
    if rc != 0 or not n730_path.exists():
        err("Conversion failed.")
        return None
    ok(f"Converted model written to {n730_path}")
    return n730_path


def stage_infer(model_id: str, n730_path: Path, prefetch: int):
    step(3, 3, "Launching interactive inference")
    core_ok, cuda_ok = check_cuda_core()
    if not (core_ok and cuda_ok):
        err("Native libraries missing — inference needs the C++ core and CUDA kernels built.")
        print()
        print(build_hint())
        return
    rc = run([
        sys.executable, str(INFERENCE),
        "--model", str(n730_path),
        "--hf-model", model_id,
        "--prefetch", str(prefetch),
        "--interactive",
    ])
    if rc != 0:
        err("Inference exited with an error.")


# ─────────────────────────────────────────────────────────────────────────
#  Menu actions
# ─────────────────────────────────────────────────────────────────────────

def action_full_pipeline(state):
    model_id = choose_model(state)
    if not model_id:
        return
    clear(); banner()
    info(f"Model     : {model_id}")
    info(f"HF_HOME   : {os.environ['HF_HOME']}")
    rule()

    sensitivity_path = stage_profile(model_id)
    if not sensitivity_path:
        pause(); return

    n730_path = stage_convert(model_id, sensitivity_path)
    if not n730_path:
        pause(); return

    state["last_model"] = model_id
    state["last_n730"] = str(n730_path)
    save_state(state)

    if prompt_yesno("Run interactive inference now?", default=True):
        prefetch = prompt_text("Prefetch depth (layers to stage ahead)", default="8")
        try:
            prefetch = int(prefetch)
        except (TypeError, ValueError):
            prefetch = 8
        stage_infer(model_id, n730_path, prefetch)

    pause()


def action_profile_only(state):
    model_id = choose_model(state)
    if not model_id:
        return
    clear(); banner()
    stage_profile(model_id)
    pause()


def action_convert_only(state):
    model_id = choose_model(state)
    if not model_id:
        return
    run_dir = MODELS_DIR / slug(model_id)
    default_map = run_dir / "sensitivity_map.json"
    sensitivity = prompt_text(
        "Path to sensitivity_map.json",
        default=str(default_map) if default_map.exists() else None,
    )
    if not sensitivity or not Path(sensitivity).exists():
        err("Sensitivity map not found — run profiling first.")
        pause(); return
    clear(); banner()
    stage_convert(model_id, Path(sensitivity))
    pause()


def action_inference_only(state):
    last_n730 = state.get("last_n730")
    n730_path = prompt_text("Path to .n730 file", default=last_n730)
    if not n730_path or not Path(n730_path).exists():
        err(".n730 file not found.")
        pause(); return
    model_id = choose_model(state)
    if not model_id:
        return
    prefetch = prompt_text("Prefetch depth", default="8")
    try:
        prefetch = int(prefetch)
    except (TypeError, ValueError):
        prefetch = 8
    clear(); banner()
    stage_infer(model_id, Path(n730_path), prefetch)
    pause()


def action_check_env(state):
    clear(); banner()
    print(f"{C.BOLD}Environment check{C.RESET}")
    rule()
    info(f"Python       : {platform.python_version()} ({sys.executable})")
    info(f"Platform     : {platform.platform()}")
    info(f"HF_HOME      : {os.environ.get('HF_HOME', '(not set)')}")

    missing = check_python_packages()
    if missing:
        warn(f"Missing packages: {', '.join(missing)}")
        if prompt_yesno("Install them now with pip?", default=True):
            install_packages(missing)
    else:
        ok("All required Python packages present (numpy, torch, transformers, huggingface_hub)")

    core_ok, cuda_ok = check_cuda_core()
    if core_ok:
        ok(f"C++ scheduler core found: {CORE_LIB}")
    else:
        warn(f"C++ scheduler core NOT found: {CORE_LIB}")
    if cuda_ok:
        ok(f"CUDA kernel library found: {CUDA_LIB}")
    else:
        warn(f"CUDA kernel library NOT found: {CUDA_LIB}")
    if not (core_ok and cuda_ok):
        print()
        print(build_hint())

    pause()


def action_inspect_n730(state):
    last_n730 = state.get("last_n730")
    n730_path = prompt_text("Path to .n730 file to inspect", default=last_n730)
    if not n730_path or not Path(n730_path).exists():
        err(".n730 file not found.")
        pause(); return
    clear(); banner()
    run([sys.executable, str(CONVERTER), "--inspect", n730_path])
    pause()


# ─────────────────────────────────────────────────────────────────────────
#  Main menu
# ─────────────────────────────────────────────────────────────────────────

def main():
    if sys.version_info < (3, 9):
        print("N730 setup requires Python 3.9+.")
        sys.exit(1)

    state = load_state()
    hf_home = setup_hf_home()

    while True:
        clear()
        banner()
        info(f"HF_HOME set to {hf_home}")
        if state.get("last_model"):
            info(f"Last model used: {state['last_model']}")
        rule()

        options = [
            ("Full pipeline (profile → convert → chat)", "Recommended for a first run"),
            ("Profile a model only", "Writes sensitivity_map.json"),
            ("Convert a profiled model only", "Needs an existing sensitivity_map.json"),
            ("Run inference only", "Needs an existing .n730 file"),
            ("Inspect a .n730 file", "Prints header + seek table summary"),
            ("Check environment / build native libraries", "Python packages + CUDA/C++ status"),
            ("Quit", None),
        ]
        idx = prompt_choice("What would you like to do?", options, allow_back=False)

        actions = [
            action_full_pipeline,
            action_profile_only,
            action_convert_only,
            action_inference_only,
            action_inspect_n730,
            action_check_env,
        ]

        if idx == len(actions):
            break
        try:
            actions[idx](state)
        except KeyboardInterrupt:
            print()
            warn("Interrupted.")
            pause()
        except Exception as e:  # noqa: BLE001 - surface any stage error, keep menu alive
            err(f"Unexpected error: {e}")
            pause()

    print(f"\n{C.DIM}  o7{C.RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Bye.")