"""CACTI 7 SRAM table: per-access energy (pJ) and area (mm2) by (size bytes, width bytes)."""

import csv
from . import config


class Cacti:
    def __init__(self, folder=config.DATA / "cacti", tech=config.TECH_UM):
        self.rows = {}
        for path in sorted(folder.glob("*.csv")):
            for r in csv.DictReader(open(path)):
                if r["ok"] == "1" and r["tech_um"] == tech:
                    key = (int(r["size_B"]), int(r["width_B"]))
                    self.rows[key] = (
                        float(r["read_nJ"]) * 1e3,
                        float(r["write_nJ"]) * 1e3,
                        float(r["area_mm2"]),
                    )

    def read(self, size, width):
        return self.rows[(size, width)][0]

    def write(self, size, width):
        return self.rows[(size, width)][1]

    def area(self, size, width):
        return self.rows[(size, width)][2]
