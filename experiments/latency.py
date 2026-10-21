"""Forward-pass latency of the four generator variants (full width, depth 5) on the machine that runs the script.

    python experiments/latency.py --patch 64 --repeats 5 --out latency.json

The numbers describe THIS machine (CPU or GPU, stated in the output) and are not a property of the models.
"""
import argparse, json, platform, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from bratsgan.models import UNet3D

VARIANTS = [("A0 plain", dict(residual=False, attention=False, fusion="early")),
            ("A1 + residual", dict(residual=True, attention=False, fusion="early")),
            ("A2 + attention", dict(residual=True, attention=True, fusion="early")),
            ("A3 + MAF", dict(residual=True, attention=True, fusion="maf"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="latency.json")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 4, a.patch, a.patch, a.patch, device=dev)
    rows = []
    for name, kw in VARIANTS:
        m = UNet3D(base=32, depth=5, max_ch=320, deep_supervision=False, **kw).to(dev).eval()
        with torch.no_grad():
            m(x)  # warm-up
            ts = []
            for _ in range(a.repeats):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter(); m(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        rows.append({"variant": name, "median_s": sorted(ts)[len(ts) // 2], "min_s": min(ts), "max_s": max(ts)})
        print(rows[-1], flush=True)
    info = {"device": torch.cuda.get_device_name(0) if dev.type == "cuda" else f"CPU, {torch.get_num_threads()} threads ({platform.processor() or platform.machine()})",
            "torch": torch.__version__, "patch": a.patch, "repeats": a.repeats, "rows": rows}
    Path(a.out).write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
