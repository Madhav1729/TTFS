"""The compared designs: measured logic power, clock and area, read from one source (see config.SOURCES)."""

from dataclasses import dataclass
from statistics import mean

from . import config, dc, memory


@dataclass
class Reports:
    """Paths of one source: simulations, power reports and synthesis reports."""

    paths: dict

    def sims(self, workload, engine, layer):
        return self.paths["sims"] / "sims" / workload / engine / f"{layer}.json"

    def power(self, piece, tag, sample=None):
        name = tag if sample is None else f"{tag}_u{sample}"
        return self.paths["power"] / "power" / piece / f"{name}.rpt"

    def table_area_um2(self):
        return float((self.paths["power"] / "power" / "spf_table" / "area.txt").read_text().split()[0])

    def slack(self, piece):
        return dc.worst_slack(self.paths["synth"] / "synth" / piece / "timing.rpt")

    def area(self, piece, name):
        return dc.area(self.paths["synth"] / "synth" / piece / "area.rpt", name)

    def area_matching(self, piece, pattern):
        return dc.areas_matching(self.paths["synth"] / "synth" / piece / "area.rpt", pattern)

    def mean_block(self, piece, tag, samples):
        top, drop = config.TOPS[piece], config.SRAM_BLOCKS.get(piece, [])
        powers = [dc.block_power(self.power(piece, tag, i), top, drop) for i in samples]
        return dc.Power(mean(p.dynamic for p in powers), mean(p.leakage for p in powers))


@dataclass
class SparTen:
    """SparTen W4/A4.  fair=True prices the collector's threshold table as SRAM, as TTFS's table is."""

    reports: Reports
    fair: bool = False
    engine = "sparten"

    @property
    def name(self):
        return "SparTen W4/A4 (table as SRAM)" if self.fair else "SparTen W4/A4 (as synthesized)"

    def power(self, tag):
        r = self.reports
        units = r.mean_block("spf_unit", tag, config.UNIT_SAMPLES).scale(config.UNITS)
        peri = dc.block_power(r.power("spf_peri", tag), "spf_periphery", config.SRAM_BLOCKS["spf_peri"])
        if not self.fair:
            return units + peri
        return units + peri - dc.cell_totals(r.power("spf_table", tag))

    def stats(self, result):
        return result["engines"][self.engine]

    def memory_pJ(self, stats, cacti):
        e = memory.sparten(stats, cacti)
        if self.fair:
            e["table"] = memory.sparten_table(stats, cacti)
        return e

    def clock_ns(self):
        return config.PERIOD_NS - min(0.0, self.reports.slack("spf_peri"))

    def area_mm2(self, cacti):
        r = self.reports
        unit = r.area("spf_unit", "spf_unit") - r.area("spf_unit", "fbuf") - r.area("spf_unit", "aread")
        peri = r.area("spf_peri", "spf_periphery") - r.area_matching("spf_peri", r"homes_\d+__outsums")
        logic = config.UNITS * unit + peri
        unit_buffers = cacti.area(512, 16) + cacti.area(256, 8) + cacti.area(512, 8)
        sram = config.UNITS * unit_buffers + cacti.area(2048, 8)
        if self.fair:
            logic -= r.table_area_um2()
            sram += cacti.area(2048, 16)
        return logic * 1e-6 + sram


@dataclass
class TTFS:
    """TTFS-64 with S/U lanes (ttfs_iv).  gated=True uses the idle-lane clock-gated lane."""

    reports: Reports
    gated: bool = False
    engine = "ttfs"

    @property
    def lane(self):
        return "iv_lane_gated" if self.gated else "iv_lane"

    @property
    def name(self):
        return "TTFS-64 S/U, gated lanes" if self.gated else "TTFS-64 S/U"

    def power(self, tag):
        r = self.reports
        lanes = r.mean_block(self.lane, tag, config.LANE_SAMPLES).scale(config.LANES)
        front = dc.block_power(r.power("iv_front", tag), "ttfs_iv", config.SRAM_BLOCKS["iv_front"])
        return lanes + front

    def stats(self, result):
        return result["engines"][self.engine]

    def memory_pJ(self, stats, cacti):
        return memory.ttfs(stats, cacti)

    def clock_ns(self):
        return config.PERIOD_NS - min(0.0, self.reports.slack(self.lane), self.reports.slack("iv_front"))

    def area_mm2(self, cacti):
        r = self.reports
        storage = sum(r.area("iv_front", b) for b in config.SRAM_BLOCKS["iv_front"])
        front = r.area("iv_front", "ttfs_iv") - storage
        logic = config.LANES * r.area(self.lane, "ttfs_iv_lane") + front
        sram = 33 * cacti.area(512, 4) + cacti.area(16384, 64) + cacti.area(2048, 16) + cacti.area(4096, 8)
        return logic * 1e-6 + sram


@dataclass
class APEX:
    """APEX: 16 INT4 TPPEs (LoAS) + PASC-IF periphery, rate code T = 15."""

    reports: Reports
    engine = "apex"
    name = "APEX (T=15, 16 INT4), measured"

    def power(self, tag):
        r = self.reports
        tppes = r.mean_block("apex_tppe", tag, config.TPPE_SAMPLES).scale(config.TPPES)
        return tppes + dc.block_power(r.power("apex_peri", tag), "apex_periphery")

    def stats(self, result):
        return result

    def memory_pJ(self, stats, cacti):
        return memory.apex(stats, cacti)

    def clock_ns(self):
        return config.PERIOD_NS - min(0.0, self.reports.slack("apex_tppe"), self.reports.slack("apex_peri"))

    def area_mm2(self, cacti):
        r = self.reports
        tppe = r.area("apex_tppe", "apex_tppe") - sum(
            r.area("apex_tppe", b) for b in config.SRAM_BLOCKS["apex_tppe"]
        )
        logic = config.TPPES * tppe + r.area("apex_peri", "apex_periphery")
        sram = config.TPPES * (cacti.area(64, 2) + 2 * cacti.area(256, 4))
        return logic * 1e-6 + sram


def all_designs(source):
    reports = Reports(config.SOURCES[source])
    return [SparTen(reports), SparTen(reports, fair=True), TTFS(reports), TTFS(reports, gated=True)]
