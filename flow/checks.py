"""Report-by-report comparison of regenerated power reports with the recorded ones."""

from . import config, dc

TOP_AND_DROP = {
    "iv_lane": ("ttfs_iv_lane", []),
    "iv_lane_gated": ("ttfs_iv_lane", []),
    "iv_front": ("ttfs_iv", config.SRAM_BLOCKS["iv_front"]),
    "spf_unit": ("spf_unit", config.SRAM_BLOCKS["spf_unit"]),
    "spf_peri": ("spf_periphery", config.SRAM_BLOCKS["spf_peri"]),
    "spf_table": (None, []),
}


def _relative(a, b):
    return max(
        abs(a.dynamic - b.dynamic) / max(b.dynamic, 1e-12), abs(a.leakage - b.leakage) / max(b.leakage, 1e-12)
    )


def power_reports(piece, root):
    """(reports compared, worst relative difference of dynamic or leakage power) vs data/recorded."""
    top, drop = TOP_AND_DROP[piece]
    new = sorted((root / "power" / piece).glob("*.rpt"))
    worst = 0.0
    for path in new:
        recorded = config.RECORDED / "power" / piece / path.name
        if piece == "spf_table":
            worst = max(worst, _relative(dc.cell_totals(path), dc.cell_totals(recorded)))
        else:
            worst = max(
                worst, _relative(dc.block_power(path, top, drop), dc.block_power(recorded, top, drop))
            )
    return len(new), worst
