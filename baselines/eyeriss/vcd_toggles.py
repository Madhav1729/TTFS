#!/usr/bin/env python3
"""Bit toggles per cycle of the Eyeriss PE in a VCD, outside the filter spad."""

import json
import re
import sys

DATA = re.compile(
    r"^(acc_val|prod2|pin1|pin2|pin_beat|pin_reg|pspad|ostage|oq_data|pq_data|a0|a1|a_hold|fs_rdata|po_data|pi_data)"
)


def toggles(path):
    ids = {}  # id -> (width, group), for ids seen in a counted scope
    fspad_only = set()
    scope = []
    val = {}
    tog = {"data": 0, "rest": 0}
    t_last = 0
    with open(path) as f:
        for line in f:
            if line.startswith("$enddefinitions") or "$enddefinitions" in line:
                break
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "$scope":
                scope.append(tok[2])
            elif tok[0] == "$upscope":
                scope.pop()
            elif tok[0] == "$var":
                width, code, name = int(tok[2]), tok[3], tok[4]
                if name == "clk" or len(scope) < 2:
                    continue
                if "fspad" in scope:
                    fspad_only.add(code)
                    continue
                data = DATA.search(name) or (scope[-1] in ("fifo_pi", "fifo_po") and name.startswith("mem"))
                ids.setdefault(code, (width, "data" if data else "rest"))
        for line in f:
            c = line[0]
            if c == "#":
                t_last = int(line[1:])
                continue
            if c == "b":
                v, code = line[1:].split()
            elif c in "01xz":
                v, code = c, line[1:].strip()
            else:
                continue
            if code not in ids:
                continue
            width, grp = ids[code]
            v = int(v.replace("x", "0").replace("z", "0"), 2)
            old = val.get(code)
            if old is not None:
                tog[grp] += bin(old ^ v).count("1")
            val[code] = v
    cycles = t_last // 5000 + 1  # 5 ns clock, 1 ps timescale (pe_tb.cpp)
    return dict(
        file=path,
        cycles=cycles,
        signals=len(ids),
        bits=sum(w for w, _ in ids.values()),
        toggles_data=tog["data"],
        toggles_rest=tog["rest"],
        rate_data=tog["data"] / cycles,
        rate_rest=tog["rest"] / cycles,
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(toggles(p)))
