"""Readers for Design Compiler reports (power, area, timing)."""

import re
from dataclasses import dataclass
from pathlib import Path

UNIT_MW = {"mW": 1.0, "uW": 1e-3, "nW": 1e-6, "pW": 1e-9}
HIER_ROW = re.compile(
    r"^(\s*)(\S+)(?:\s+\(\S+\))?\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)\s*$"
)
TOTALS_ROW = re.compile(r"Totals \(\d+ cells\)\s+([\d.]+)(\w+)\s+N/A\s+N/A \(N/A\)\s+([\d.]+)(\w+)")


@dataclass
class Power:
    dynamic: float = 0.0
    leakage: float = 0.0

    def __add__(self, other):
        return Power(self.dynamic + other.dynamic, self.leakage + other.leakage)

    def __sub__(self, other):
        return Power(self.dynamic - other.dynamic, self.leakage - other.leakage)

    def scale(self, k):
        return Power(self.dynamic * k, self.leakage * k)

    def energy_pJ(self, cycles, clock_ns, period_ns):
        """Dynamic power is per toggle of the period it was recorded at; leakage runs for the real clock."""
        return cycles * (self.dynamic * period_ns + self.leakage * clock_ns)


def _units(text):
    dyn = re.search(r"Dynamic Power Units = 1(\w+)", text).group(1)
    leak = re.search(r"Leakage Power Units = 1(\w+)", text).group(1)
    return UNIT_MW[dyn], UNIT_MW[leak]


def hierarchy(path):
    """{instance path: Power} from report_power -hierarchy."""
    text = Path(path).read_text()
    dyn_unit, leak_unit = _units(text)
    out, stack = {}, []
    for line in text.splitlines():
        m = HIER_ROW.match(line)
        if not m:
            continue
        stack = stack[: len(m.group(1)) // 2] + [m.group(2)]
        total, leak = float(m.group(6)) * dyn_unit, float(m.group(5)) * leak_unit
        out["/".join(stack)] = Power(total - leak, leak)
    return out


def block_power(path, top, drop=()):
    """Power of `top` minus its direct children matching any regex in `drop`."""
    rep = hierarchy(path)
    power = rep[top]
    for name, p in rep.items():
        parts = name.split("/")
        if len(parts) == 2 and any(re.fullmatch(d, parts[1]) for d in drop):
            power = power - p
    return power


def cell_totals(path):
    """Power of a report_power -cell -only selection (its Totals row)."""
    m = TOTALS_ROW.search(Path(path).read_text())
    return Power(float(m.group(1)) * UNIT_MW[m.group(2)], float(m.group(3)) * UNIT_MW[m.group(4)])


def area(path, name):
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == name:
            return float(fields[1])
    raise KeyError(f"{name} not in {path}")


def areas_matching(path, pattern):
    text = Path(path).read_text()
    return sum(float(x) for x in re.findall(rf"^{pattern}\s+([\d.]+)", text, re.M))


def worst_slack(path):
    return min(
        float(x) for x in re.findall(r"slack \((?:MET|VIOLATED)\)\s+([-\d.]+)", Path(path).read_text())
    )
