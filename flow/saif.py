"""Make Verilator SAIF readable by Design Compiler and cut out one instance's subtree."""

import re

TIMED = re.compile(r"\((T0|T1|TZ|TX|TB) (\d+)\)")
IDENT = re.compile(r"^(\s*\((?:INSTANCE\s+)?)(\S+)(.*)$")
KEYWORDS = {"NET", "PORT", "INSTANCE"}


def _escape(name):
    return re.sub(r"(?<!\\)([\[\]])", r"\\\1", name)


def _fix_line(line, scale):
    if scale != 1.0:
        line = TIMED.sub(lambda m: f"({m.group(1)} {round(int(m.group(2)) * scale)})", line)
    m = IDENT.match(line)
    if not m or m.group(2) in KEYWORDS:
        return line
    return m.group(1) + _escape(m.group(2)) + m.group(3) + ("\n" if line.endswith("\n") else "")


def _split(lines):
    head = {}
    for k, line in enumerate(lines):
        s = line.strip()
        if s.startswith("(INSTANCE"):
            return head, lines[k:]
        m = re.match(r"\((\w+)\s*(.*)\)$", s)
        if m:
            head[m.group(1)] = m.group(2).strip()
    raise ValueError("no INSTANCE in SAIF")


def _header(head, scale):
    unit = re.match(r"(\d+)\s*(\w+)", head["TIMESCALE"])
    return (
        '(SAIFILE\n(SAIFVERSION "2.0")\n(DIRECTION "backward")\n(DESIGN )\n(DATE "unknown")\n'
        '(VENDOR "Verilator")\n(PROGRAM_NAME "Verilator")\n(VERSION "5")\n(DIVIDER / )\n'
        f"(TIMESCALE {unit.group(1)} {unit.group(2)})\n(DURATION {round(int(head['DURATION']) * scale)})\n"
    )


def _subtree(body, path):
    want = [_escape(p) for p in path.split("/")]
    stack, depth, start = [], 0, None
    for k, line in enumerate(body):
        m = re.match(r"^\s*\(INSTANCE\s+(\S+)", line)
        if m:
            stack.append((_escape(m.group(1)), depth))
            if start is None and [s for s, _ in stack] == want:
                start, start_depth = k, depth
        depth += line.count("(") - line.count(")")
        while stack and depth <= stack[-1][1]:
            stack.pop()
        if start is not None and depth <= start_depth:
            return body[start : k + 1]
    raise KeyError(f"instance {path} not in SAIF")


def extract(src, dst, path, top, scale):
    """Write the subtree at `path` of `src` as a SAIF whose top instance is `top`, durations x scale."""
    head, body = _split(open(src).read().splitlines(keepends=True))
    sub = _subtree(body, path)
    sub[0] = re.sub(r"\(INSTANCE\s+\S+", f"(INSTANCE {top}", sub[0], count=1)
    with open(dst, "w") as out:
        out.write(_header(head, scale))
        out.writelines(_fix_line(line, scale) for line in sub)
        out.write(")\n")
