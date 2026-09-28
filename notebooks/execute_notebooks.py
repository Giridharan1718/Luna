"""Execute .ipynb notebooks headlessly, saving outputs into the notebook files.

Usage:
    python notebooks/execute_notebooks.py 01 02 03     # by number prefix
    python notebooks/execute_notebooks.py all
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from jupyter_client.manager import start_new_kernel

HERE = Path(__file__).parent


def run_notebook(nb_path: Path, timeout: int = 3600) -> tuple[bool, str]:
    nb = nbformat.read(str(nb_path), as_version=4)
    km, kc = start_new_kernel(kernel_name="python3")
    errors: list[str] = []
    try:
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            msg_id = kc.execute(cell.source)
            while True:
                msg = kc.get_iopub_msg(timeout=timeout)
                msg_type = msg["header"]["msg_type"]
                content = msg["content"]
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                if msg_type == "stream":
                    text = content.get("text", "")
                    print(f"    | {text.strip()[:400]}")
                elif msg_type in ("error",):
                    ename = content.get("ename", "Error")
                    evalue = content.get("evalue", "")
                    tb = "\n".join(content.get("traceback", [])[-6:])
                    errors.append(f"[cell {i}] {ename}: {evalue}\n{tb}")
                    print(f"    !! {ename}: {evalue}")
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    break
            cell["execution_count"] = i
            # attach a marker output summarizing success for this cell
            if not any(e.startswith(f"[cell {i}]") for e in errors):
                cell.setdefault("outputs", []).append(
                    nbformat.v4.new_output("stream", name="stdout",
                                           text=f"[lunarai] cell {i} executed ok"))
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    nbformat.write(nb, str(nb_path))
    return (len(errors) == 0, "\n".join(errors))


def main() -> int:
    args = sys.argv[1:] or ["all"]
    targets = []
    for nb in sorted(HERE.glob("nb*.ipynb")):
        num = nb.stem.split("_")[0].replace("nb", "")   # e.g. "01"
        if "all" in args or num in args or nb.stem in args:
            targets.append(nb)
    if not targets:
        print("no notebooks matched:", args)
        return 1
    failures = []
    for nb in targets:
        print(f"\n=== executing {nb.name} ===")
        t0 = time.perf_counter()
        ok, err = run_notebook(nb)
        dt = time.perf_counter() - t0
        print(f"=== {nb.name}: {'OK' if ok else 'FAILED'} ({dt:.1f}s) ===")
        if not ok:
            failures.append((nb.name, err))
    if failures:
        print("\nFAILURES:")
        for name, err in failures:
            print(f"\n--- {name} ---\n{err[-3000:]}")
        return 2
    print("\nall notebooks executed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
