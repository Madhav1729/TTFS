"""Command line:  python -m flow <step> ...   (run from hw_final/)"""

import argparse
import json
from pathlib import Path

from . import build, checks, config, dcjobs, export, lm, report, simulate

POWER_PIECES = list(dcjobs.PIECES)


def cmd_build(a):
    for gate in (0, 1):
        print(build.build(gate))
    print(build.build(1, steps=8))
    print(build.build_apex())
    print(build.build_eyeriss())


def cmd_simulate(a):
    for engine in ("ttfs", "sparten"):
        runs = simulate.full_layers(engine, a.jobs)
        print(f"full layers: {len(runs)} {engine} runs verified")
    for gate, engine in ((0, "ttfs"), (1, "ttfs"), (0, "sparten")):
        runs = simulate.activity(gate, engine, a.jobs)
        print(f"activity: {len(runs)} {engine} runs verified (gate {gate})")
    print(
        f"APEX: {len(simulate.apex_full_layers(a.jobs))} full-layer and "
        f"{len(simulate.apex_activity(a.jobs))} activity runs exact"
    )
    print(f"Eyeriss PE: {len(simulate.eyeriss_activity(a.jobs))} activity runs exact")


def cmd_check_gating(a):
    runs = {g: simulate.activity(g, "ttfs", a.jobs) for g in (0, 1)}
    same = simulate.same_schedule(runs[0], runs[1])
    print(f"{len(runs[0])} layers: gating {'keeps' if same else 'CHANGES'} cycles and arithmetic")


def cmd_power(a):
    unknown = set(a.pieces) - set(POWER_PIECES)
    if unknown:
        raise SystemExit(f"unknown pieces: {sorted(unknown)}")
    for piece in a.pieces or POWER_PIECES:
        print(piece, dcjobs.power(piece, a.netlists))


def cmd_synth(a):
    for piece in a.pieces or list(config.SYNTH_PIECES):
        print(piece, dcjobs.synth(piece))


def cmd_fetch(a):
    for label in a.labels or list(dcjobs.launched()):
        print(label, dcjobs.fetch(label))


def cmd_table(a):
    rows = report.build_rows(a.source)
    print(report.write(rows, "final" if a.source == "recorded" else "final_rerun"))


def cmd_lm(a):
    if not a.reuse:
        lm.make_problems(a.weights)
        lm.simulate_all(a.weights, a.jobs)
    print(lm.write_table(a.weights))


def cmd_export_problems(a):
    print(export.export(a.density, a.out or config.RESULTS / "problems" / f"w_{a.density}", image=a.image))


def cmd_quantize_otters(a):
    print("zero fraction per matrix:", lm.quantize_otters())


def cmd_check_reports(a):
    for root, label in ((config.RESULTS, "existing netlists"), (config.RESULTS / "resynth", "resynthesized")):
        for piece in checks.TOP_AND_DROP:
            if (root / "power" / piece).is_dir():
                n, worst = checks.power_reports(piece, root)
                print(f"{label:18s} {piece:14s} {n:4d} reports, worst relative difference {worst:.2e}")


def cmd_compare(a):
    rerun = report.build_rows(a.source)
    recorded = json.loads((config.RESULTS / "final.json").read_text())
    diffs = report.compare(rerun, recorded, a.tolerance)
    for d in diffs:
        print("DIFF", *d)
    print(f"{len(rerun)} rows compared, {len(diffs)} differences (relative tolerance {a.tolerance})")


def main():
    ap = argparse.ArgumentParser(prog="python -m flow")
    sub = ap.add_subparsers(required=True)

    sub.add_parser(
        "build", help="Verilator builds: TTFS / SparTen (both lanes, T = 15 and 8), APEX, Eyeriss"
    ).set_defaults(func=cmd_build)

    p = sub.add_parser("simulate", help="full-layer runs and SAIF activity runs")
    p.add_argument("--jobs", type=int, default=8)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("check-gating", help="gated vs plain lane: same cycles and arithmetic")
    p.add_argument("--jobs", type=int, default=8)
    p.set_defaults(func=cmd_check_gating)

    p = sub.add_parser("power", help="launch power-only DC runs (default: all pieces)")
    p.add_argument("pieces", nargs="*", help=f"any of {', '.join(POWER_PIECES)}")
    p.add_argument("--netlists", default="data", choices=("data", "resynth"))
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("synth", help="synthesize pieces from rtl/ with DC (default: all)")
    p.add_argument("pieces", nargs="*", help=f"any of {', '.join(config.SYNTH_PIECES)}")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("fetch", help="copy finished DC runs into results/ (default: all launched)")
    p.add_argument("labels", nargs="*")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("table", help="final table from recorded reports or from this flow's results")
    p.add_argument("--source", default="recorded", choices=list(config.SOURCES))
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("lm", help="BERT encoder layer: activation codes and QNN baselines")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--reuse", action="store_true", help="reuse existing simulations")
    p.add_argument("--weights", default="w1", choices=("w1", "w4"), help="binary (Otters) or 4-bit weights")
    p.set_defaults(func=cmd_lm)

    p = sub.add_parser("export-problems", help="VGG-16 layer exports (models/vgg16/layers) -> problem files")
    p.add_argument("--density", default="natural", help="natural, a weight density (0.6, 0.3) or loas")
    p.add_argument("--image", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_export_problems)

    sub.add_parser("quantize-otters", help="Otters layer-5 weights -> data/lm (4-bit)").set_defaults(
        func=cmd_quantize_otters
    )

    sub.add_parser("check-reports", help="regenerated power reports vs recorded").set_defaults(
        func=cmd_check_reports
    )

    p = sub.add_parser("compare", help="a rerun table against the recorded table")
    p.add_argument("--source", default="results", choices=["results", "resynth"])
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.set_defaults(func=cmd_compare)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
