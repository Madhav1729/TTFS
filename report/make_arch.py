"""Writes arch.drawio: TTFS-64 tile (left) and one S/U lane (right)."""

from xml.sax.saxutils import escape

cells, ids = [], {}


def box(key, x, y, w, h, label, fill, stroke="#666666", size=12, extra=""):
    ids[key] = f"n{len(ids) + 2}"
    style = (f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontFamily=Helvetica;fontSize={size};fontStyle=1;{extra}")
    cells.append(f'<mxCell id="{ids[key]}" value="{escape(label)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(a, b, exit=None, entry=None, points=(), both=False):
    style = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=1.3;"
    if both:
        style += "startArrow=block;startFill=1;"
    if exit:
        style += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
    if entry:
        style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
    pts = f'<Array as="points">{pts}</Array>' if pts else ""
    cells.append(f'<mxCell id="e{len(cells)}" style="{style}" edge="1" parent="1" source="{ids[a]}" '
                 f'target="{ids[b]}"><mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')


def line(x1, y1, x2, y2):
    cells.append(f'<mxCell id="e{len(cells)}" style="endArrow=none;html=1;dashed=1;" edge="1" parent="1">'
                 f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
                 f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')


NONE = "strokeColor=none;fillColor=none;"

# tile
box("tile", 0, 0, 520, 520, "", "#f3f3f1", "#555555", extra="arcSize=4;")
box("cmp", 40, 24, 440, 48, "Spike Compaction", "#d5ccf0", "#9673a6", 15)
lx = [55, 148, 241, 394]
for i, x in enumerate(lx):
    box(f"c{i}", x, 104, 72, 36, "Compare", "#fde3b0", "#d79b00", 11)
    box(f"l{i}", x, 166, 72, 72, "S/U Lane", "#a8835f", "#6d4c2e", 12, "fontColor=#ffffff;")
for y in (122, 202):
    box(f"dots{y}", 318, y - 10, 70, 20, "· · · · ·", "none", "none", 14, NONE)
box("win", 40, 270, 312, 52, "Event Window", "#a8e2ea", "#3a9fab", 15)
box("wb", 375, 270, 110, 52, "Weight Banks", "#fde3b0", "#d79b00", 13)
box("src", 40, 348, 440, 46, "Patch Buffer &amp; Event Source", "#bfe6dc", "#3a9f7b", 15)
box("glb", 40, 426, 440, 60, "Global Buffer (SRAM)", "#d0d0d0", "#666666", 15)
for i in range(4):
    edge(f"l{i}", f"c{i}", exit=(0.5, 0), entry=(0.5, 1))
    edge(f"c{i}", "cmp", exit=(0.5, 0), entry=((lx[i] + 36 - 40) / 440, 1))
    if i < 3:
        edge("win", f"l{i}", exit=((lx[i] + 36 - 40) / 312, 0), entry=(0.5, 1))
edge("wb", "l3", exit=(0.5, 0), entry=(0.5, 1))
edge("wb", "win", exit=(0, 0.5), entry=(1, 0.5))
edge("src", "win", exit=(0.35, 0), entry=(0.5, 1))
edge("glb", "src", exit=(0.12, 0), entry=(0.12, 1), both=True)
edge("glb", "wb", exit=(0.88, 0), entry=(0.5, 1), points=[(427, 410), (430, 410)])
edge("cmp", "glb", exit=(1, 0.5), entry=(1, 0.5), points=[(500, 48), (500, 456)])

# one lane
X = 570
box("zoom", X, 0, 530, 520, "", "#e3f4d7", "#333333", extra="arcSize=4;")
line(466, 166, X, 0)
line(466, 238, X, 520)
box("fifo", X + 30, 28, 120, 34, "Event FIFO", "#d0d0d0", "#666666", 11)
box("zwb", X + 200, 28, 136, 34, "Weight Banks", "#fbb987", "#d6732d", 11)
for i in range(4):
    box(f"wbs{i}", X + 206 + i * 33, 70, 24, 22, "", "#fde8d6", "#d6732d")
box("zwin", X + 20, 92, 140, 82, "Event Window", "#d6d6d6", "#666666", 11, "verticalAlign=top;")
for r in range(2):
    for c in range(4):
        box(f"s{r}{c}", X + 30 + c * 31, 118 + r * 26, 25, 20, "", "#f2f2f2", "#888888")
box("pend", X + 30, 200, 120, 34, "Pending Mask", "#a9c4f5", "#6c8ebf", 11)
box("pick", X + 30, 260, 120, 34, "Oldest-Pending Pick", "#3fcf8e", "#2d8f60", 11)
box("sadd", X + 30, 330, 120, 70, "S += w", "#b4c4ec", "#6c8ebf", 14, "arcSize=20;")
box("tick", X + 30, 430, 120, 34, "Tick Counter", "#a9c4f5", "#6c8ebf", 11)
box("uadd", X + 210, 330, 120, 70, "U += S", "#b4c4ec", "#6c8ebf", 14, "arcSize=20;")
box("mem", X + 380, 330, 130, 40, "Membrane SRAM", "#dbd2ef", "#9673a6", 11)
box("thr", X + 380, 222, 130, 36, "Threshold Table", "#fbb987", "#d6732d", 11)
box("comp", X + 380, 140, 130, 36, "Comparator", "#3fcf8e", "#2d8f60", 11)
box("spike", X + 405, 52, 80, 24, "Spike", "none", "none", 11, NONE)
box("gate", X + 380, 430, 120, 34, "Clock Gate", "#e0e0e0", "#666666", 11, "dashed=1;")
edge("fifo", "zwin", exit=(0.5, 1), entry=(0.5, 0))
edge("zwb", "zwin", exit=(0.5, 1), entry=(1, 0.5), points=[(X + 268, 133)])
edge("zwin", "pend", exit=(0.5, 1), entry=(0.5, 0))
edge("pend", "pick", exit=(0.5, 1), entry=(0.5, 0))
edge("pick", "sadd", exit=(0.5, 1), entry=(0.5, 0))
edge("sadd", "uadd", exit=(1, 0.5), entry=(0, 0.5))
edge("tick", "uadd", exit=(1, 0.5), entry=(0.3, 1), points=[(X + 246, 447)])
edge("uadd", "mem", exit=(1, 0.3), entry=(0, 0.5))
edge("mem", "comp", exit=(0.25, 0), entry=(0.25, 1))
edge("thr", "comp", exit=(0.75, 0), entry=(0.75, 1))
edge("comp", "spike", exit=(0.5, 0), entry=(0.5, 1))

xml = ('<mxfile host="drawio"><diagram id="ttfs64" name="TTFS-64"><mxGraphModel grid="0" page="1" pageWidth="1120" '
       'pageHeight="540" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells)
       + "</root></mxGraphModel></diagram></mxfile>")
open("arch.drawio", "w").write(xml)
