"""On-chip memory energy (pJ) of one layer run, from the simulator's access counts and CACTI."""

from . import config

GB = config.GLOBAL_BUFFER_B
U = config.UNITS


def _patch(s, cacti, engine):
    size = config.PATCH_BYTES[engine]
    return s["patch_beats"] * cacti.write(size, 8) + s["local_read_bits"] / 64 * cacti.read(size, 8)


def _global(s, cacti, input_beats):
    return {
        "glb_w": s["w_beats"] * cacti.read(GB, 8),
        "glb_in": input_beats * cacti.read(GB, 8),
        "glb_out": s["output_beats"] * cacti.write(GB, 8),
    }


def threshold_table(writes_rows, compared_pixels, cacti, steps=config.STEPS):
    """Shared table: one row of 64 x 16 b (128 B, 8 words of 16 B) per tick; 2 KB at 15 ticks, 1 KB at 8."""
    size = 1024 if steps <= 8 else 2048
    return 8 * (writes_rows * cacti.write(size, 16) + compared_pixels * steps * cacti.read(size, 16))


def sparten(s, cacti):
    in_beats = s["a_beats"] - s["output_beats"] - s["patch_beats"] + s["local_beats"]
    in_value_bytes = max(0.0, (in_beats - 2 * s["input_writes"]) * 8)
    filters = (
        s["filter_mask_writes"] * cacti.write(512, 16)
        + s["filter_value_writes"] * cacti.write(512, 1)
        + s["filter_mask_reads"] * cacti.read(512, 16)
        + s["weight_reads"] * cacti.read(512, 1)
    )
    input_writes = s["input_writes"] * 2 + in_value_bytes / 8
    inputs = U * input_writes * cacti.write(256, 8) + s["arithmetic"] * cacti.read(256, 1)
    moves = s["net_moves"] * (cacti.read(512, 4) + cacti.write(512, 4))
    sums = moves + s["collector_pixels"] * U * cacti.read(512, 8)
    e = {"fbuf": filters, "ibuf": inputs, "sums": sums, "patch": _patch(s, cacti, "sparten")}
    e.update(_global(s, cacti, s["a_beats"] - s["output_beats"]))
    return e


def sparten_table(s, cacti):
    """SparTen's collector table priced like TTFS's (threshold writes are per entry, 64 per row)."""
    return threshold_table(s["threshold_writes"] / config.LANES, s["collector_pixels"], cacti)


def ttfs(s, cacti, steps=config.STEPS):
    banks = s["column_group_writes"] * cacti.write(512, 4) + s["column_group_reads"] * cacti.read(512, 4)
    membranes = 4 * (
        s["membrane_reads"] * cacti.read(16384, 64) + s["membrane_writes"] * cacti.write(16384, 64)
    )
    e = {
        "wbanks": banks,
        "membranes": membranes,
        "patch": _patch(s, cacti, "ttfs"),
        "sorter": s["event_pushes"] * (cacti.write(512, 4) + cacti.read(512, 4)),
        "thresholds": threshold_table(s["threshold_writes"], s["compare_pixels"], cacti, steps),
    }
    e.update(_global(s, cacti, s["a_beats"] - s["output_beats"]))
    return e


def apex(s, cacti):
    """APEX: data-B (64 B per TPPE), data-A (2 x 256 B banks per TPPE), global cache in 16 KB banks."""
    tppes = config.TPPES
    return {
        "data_b": s["b_data_beats"] * tppes * 4 * cacti.write(64, 2) + s["matches"] * cacti.read(64, 1),
        "data_a": s["a_data_beats"] * 2 * cacti.write(256, 4) + s["da_row_reads"] * cacti.read(256, 4),
        "glb_read": s["gc_read_beats"] * cacti.read(16384, 8),
        "glb_write": s["gc_write_beats"] * cacti.write(16384, 8),
    }
