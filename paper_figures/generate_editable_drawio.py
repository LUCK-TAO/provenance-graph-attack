from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET


OUT = Path(__file__).with_name("handdrawn-paper-figures-editable.drawio")
INDIVIDUAL_OUTPUTS = {
    "detector-adaptation": "detector-adaptation-handdrawn-editable.drawio",
    "attack-framework": "attack-framework-handdrawn-editable.drawio",
    "candidate-construction": "candidate-node-selection-handdrawn-editable.drawio",
    "macro-action-feedback": "macro-action-feedback-handdrawn-editable.drawio",
}


FONT = "Comic Sans MS"
INK = "#25282C"
BLUE = "#74C0FC"
GREEN = "#8CE99A"
YELLOW = "#FFE066"
CORAL = "#FF8787"
PURPLE = "#B197FC"


def join_style(*parts: str) -> str:
    return ";".join(p.strip(";") for p in parts if p) + ";"


BOX = join_style(
    "rounded=1", "whiteSpace=wrap", "html=1", "sketch=1", "curveFitting=1",
    "jiggle=2", f"strokeColor={INK}", "strokeWidth=2", "fillColor=#FFFFFF",
    f"fontFamily={FONT}", "fontSize=15", f"fontColor={INK}", "align=center",
    "verticalAlign=middle", "spacing=8",
)
TEXT = join_style(
    "text", "html=1", "whiteSpace=wrap", "overflow=hidden", "fillColor=none",
    "strokeColor=none", f"fontFamily={FONT}", "fontSize=15", f"fontColor={INK}",
    "align=center", "verticalAlign=middle",
)
TITLE = join_style(TEXT, "fontSize=26", "fontStyle=1")
STAGE = join_style(TEXT, "fontSize=20", "fontStyle=1", "align=left")
SMALL = join_style(TEXT, "fontSize=12")
EDGE = join_style(
    "edgeStyle=orthogonalEdgeStyle", "rounded=1", "orthogonalLoop=1", "jettySize=auto",
    "html=1", "sketch=1", "curveFitting=1", "jiggle=2", f"strokeColor={INK}",
    "strokeWidth=3", "endArrow=block", "endFill=1", f"fontFamily={FONT}",
    "fontSize=12", f"fontColor={INK}", "labelBackgroundColor=#FFFDF9",
)
DASH_EDGE = join_style(EDGE, "dashed=1", "dashPattern=6 5", "strokeColor=#3B78A0", "strokeWidth=2")


class Page:
    def __init__(self, mxfile: ET.Element, page_id: str, name: str, width=1600, height=900):
        diagram = ET.SubElement(mxfile, "diagram", {"id": page_id, "name": name})
        self.model = ET.SubElement(diagram, "mxGraphModel", {
            "dx": "1422", "dy": "794", "grid": "1", "gridSize": "10", "guides": "1",
            "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
            "pageScale": "1", "pageWidth": str(width), "pageHeight": str(height),
            "math": "0", "shadow": "0",
        })
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.i = 2

    def _id(self, prefix="c"):
        value = f"{prefix}{self.i}"
        self.i += 1
        return value

    def vertex(self, value, x, y, w, h, style=BOX, parent="1", cell_id=None):
        cell_id = cell_id or self._id()
        cell = ET.SubElement(self.root, "mxCell", {
            "id": cell_id, "value": value, "style": style, "vertex": "1", "parent": parent,
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry",
        })
        return cell_id

    def edge(self, source, target, value="", style=EDGE, points=None, cell_id=None):
        cell_id = cell_id or self._id("e")
        cell = ET.SubElement(self.root, "mxCell", {
            "id": cell_id, "value": value, "style": style, "edge": "1", "parent": "1",
            "source": source, "target": target,
        })
        geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            arr = ET.SubElement(geo, "Array", {"as": "points"})
            for x, y in points:
                ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})
        return cell_id

    def label(self, value, x, y, w, h, style=TEXT):
        return self.vertex(value, x, y, w, h, style)

    def zone(self, x, y, w, h, color):
        return self.vertex("", x, y, w, h, join_style(
            "rounded=1", "arcSize=8", "whiteSpace=wrap", "html=1", "strokeColor=none",
            f"fillColor={color}", "opacity=90",
        ))

    def circle(self, value, x, y, size, color, font_color="#FFFFFF"):
        return self.vertex(value, x, y, size, size, join_style(
            "ellipse", "whiteSpace=wrap", "html=1", "sketch=1", "jiggle=2",
            f"fillColor={color}", f"strokeColor={INK}", "strokeWidth=2",
            f"fontFamily={FONT}", "fontSize=16", "fontStyle=1", f"fontColor={font_color}",
            "align=center", "verticalAlign=middle",
        ))

    def node(self, x, y, color="#FFFFFF", size=18):
        return self.vertex("", x, y, size, size, join_style(
            "ellipse", "whiteSpace=wrap", "html=1", "sketch=1", "jiggle=2",
            f"fillColor={color}", f"strokeColor={INK}", "strokeWidth=2",
        ))


def add_graph(page: Page, x, y, scale=1.0, target=False):
    pts = [(0, 36), (42, 0), (86, 40), (55, 78), (111, 5)]
    colors = [BLUE, "#FFFFFF", GREEN if not target else CORAL, "#FFFFFF", YELLOW]
    ids = [page.node(x + px * scale, y + py * scale, colors[i], 17 * scale) for i, (px, py) in enumerate(pts)]
    line = join_style(EDGE, "endArrow=none", "strokeWidth=2", "labelBackgroundColor=none")
    page.edge(ids[0], ids[1], style=line)
    page.edge(ids[1], ids[2], style=line)
    page.edge(ids[1], ids[3], style=line)
    page.edge(ids[2], ids[3], style=line)
    page.edge(ids[2], ids[4], style=line)
    return ids


def page_detector(mxfile):
    p = Page(mxfile, "detector-adaptation", "1 - Detector Adaptation", 1600, 900)
    p.label("Detector Adaptation and Unified Black-Box Interface", 250, 28, 1100, 50, TITLE)
    graph_zone = p.zone(45, 260, 350, 350, "#E7F5FF")
    native_zone = p.zone(510, 70, 580, 250, "#FFF0F0")
    adapter_zone = p.zone(500, 375, 600, 290, "#F8F9FA")
    iface_zone = p.zone(1205, 260, 350, 350, "#EBFBEE")
    logic_zone = p.zone(535, 720, 540, 145, "#FFF7D6")

    p.circle("1", 65, 280, 42, BLUE)
    p.label("Provenance Graph", 115, 275, 240, 45, STAGE)
    add_graph(p, 120, 370, 1.25)
    graph_tag = p.vertex("original or modified Gₜ", 95, 535, 245, 45, join_style(BOX, "fillColor=#D8EFFF"))

    p.circle("2", 530, 90, 42, CORAL)
    p.label("Native Detector", 585, 85, 430, 50, STAGE)
    chip = p.vertex("f(·)", 665, 160, 140, 90, join_style(BOX, "fillColor=#FFF1BD", "fontSize=28", "fontStyle=1"))
    native_text = p.label("detector-specific inference<br>native architecture and scoring", 830, 155, 220, 100, SMALL)

    p.label("Detector Adapter", 650, 398, 300, 50, join_style(STAGE, "align=center"))
    conv = p.vertex("Graph<br>conversion", 560, 500, 150, 85, BOX)
    query = p.vertex("Native<br>query", 725, 500, 150, 85, BOX)
    mapping = p.vertex("Output<br>mapping", 890, 500, 150, 85, BOX)
    p.edge(conv, query)
    p.edge(query, mapping)

    p.circle("3", 1225, 280, 42, GREEN)
    p.label("Unified Black-Box Interface", 1275, 275, 245, 65, STAGE)
    response = p.vertex("anomaly scores<br>alerted nodes<br>detection threshold", 1260, 390, 235, 130, join_style(BOX, "fillColor=#E8F8EA"))
    p.label("detector-independent response", 1250, 545, 255, 40, SMALL)

    p.circle("4", 555, 738, 42, YELLOW, INK)
    p.label("Shared Attack Logic", 610, 733, 400, 45, STAGE)
    target = p.vertex("Target selection", 610, 795, 135, 45, join_style(BOX, "fillColor=#FFF1BD"))
    action = p.vertex("Action decision", 760, 795, 135, 45, join_style(BOX, "fillColor=#E7F5FF"))
    update = p.vertex("Reward &amp; update", 910, 795, 135, 45, join_style(BOX, "fillColor=#EBFBEE"))

    p.edge(graph_tag, conv, "graph")
    p.edge(query, chip, "query", points=[(800, 430), (800, 320)])
    p.edge(chip, query, "native output", points=[(920, 315), (920, 430)])
    p.edge(mapping, response, "standardized feedback")
    p.edge(response, target, "observe and learn", points=[(1380, 690), (1100, 820)])
    p.edge(update, graph_tag, "next modified graph", points=[(450, 820), (240, 690)])


def page_attack(mxfile):
    p = Page(mxfile, "attack-framework", "2 - Attack Framework", 1800, 980)
    p.label("Semantic-Constrained Black-Box Attack Framework", 300, 20, 1200, 55, TITLE)
    zones = [
        (35, 95, 300, 760, "#FFF7DC"),
        (370, 95, 380, 760, "#EAF7FF"),
        (785, 95, 650, 760, "#F3EFFF"),
        (1470, 95, 295, 760, "#EDFAEF"),
    ]
    for x, y, w, h, c in zones:
        p.zone(x, y, w, h, c)

    p.circle("1", 55, 115, 40, INK)
    p.label("Attack Setup", 105, 110, 210, 45, STAGE)
    p.label("Test Provenance Graph", 70, 180, 230, 35, join_style(SMALL, "fontStyle=1"))
    add_graph(p, 115, 235, 0.85)
    detector = p.vertex("Black-Box Detector<br><b>f(·)</b>", 80, 365, 210, 90, BOX)
    p.label("Query and Edit Budgets", 70, 500, 230, 35, join_style(SMALL, "fontStyle=1"))
    bq = p.vertex("Bq<br>queries", 85, 555, 85, 65, join_style(BOX, "fillColor=#D8EFFF"))
    be = p.vertex("Be<br>edits", 200, 555, 85, 65, join_style(BOX, "fillColor=#FFE8E8"))

    p.circle("2", 390, 115, 40, INK)
    p.label("Target and State Construction", 440, 105, 285, 60, STAGE)
    obs = p.vertex("Observable anomaly scores and alerts", 420, 185, 280, 55, BOX)
    select = p.vertex("Target selection", 420, 280, 280, 115, join_style(BOX, "fillColor=#FFF1BD"))
    add_graph(p, 485, 305, 0.55, target=True)
    state = p.vertex("State encoding", 420, 430, 280, 60, BOX)
    pool = p.vertex("Candidate pool", 420, 535, 130, 105, join_style(BOX, "fillColor=#E7F5FF"))
    mask = p.vertex("Action mask", 570, 535, 130, 105, join_style(BOX, "fillColor=#EBFBEE"))

    p.circle("3", 805, 115, 40, INK)
    p.label("Semantic-Constrained Attack Loop", 855, 110, 520, 45, STAGE)
    decision = p.vertex("Action Decision<br><b>LinUCB · DDQN · PPO</b>", 950, 180, 320, 70, join_style(BOX, "fillColor=#FFF1BD"))
    macro = p.vertex("Macro-Action<br>L primitive edits", 1190, 305, 190, 85, join_style(BOX, "fillColor=#E7F5FF"))
    valid = p.vertex("Semantic Validation<br>type · event · direction", 1190, 465, 190, 85, join_style(BOX, "fillColor=#EBFBEE"))
    modify = p.vertex("Graph Modification", 950, 620, 260, 75, join_style(BOX, "fillColor=#F3EFFF"))
    query2 = p.vertex("Detector Query<br>f(·)", 835, 465, 180, 85, BOX)
    feedback = p.vertex("Feedback and Update<br>score · threshold · cost", 835, 305, 200, 90, join_style(BOX, "fillColor=#FFE8E8"))
    center = p.vertex("Iterative<br>Black-Box<br>Feedback", 1035, 395, 135, 100, join_style(
        "ellipse", "whiteSpace=wrap", "html=1", "sketch=1", "jiggle=2", "dashed=1",
        f"strokeColor={INK}", "fillColor=#FFFFFF", f"fontFamily={FONT}", "fontStyle=1",
    ))
    stop = p.vertex("Stop when the target evades<br>or the budget expires", 950, 750, 350, 65, join_style(BOX, "shape=rhombus", "fillColor=#FFFFFF"))

    p.circle("4", 1490, 115, 40, INK)
    p.label("Evaluation", 1540, 110, 200, 45, STAGE)
    metrics = p.vertex("Attack Success Rate ↑<br>Average Budget ↓<br>Average Query Time ↓", 1510, 205, 215, 140, BOX)
    adv = p.vertex("Adversarial Provenance Graph", 1510, 420, 215, 200, join_style(BOX, "fillColor=#FFFFFF"))
    add_graph(p, 1555, 485, 0.7, target=True)
    p.vertex("Attack metrics<br>▂ ▅ ▇", 1510, 670, 215, 105, join_style(BOX, "fillColor=#FFFFFF", "fontSize=20"))

    p.edge(detector, obs, "observable output")
    p.edge(pool, decision, "state and valid actions")
    p.edge(mask, decision)
    p.edge(decision, macro)
    p.edge(macro, valid)
    p.edge(valid, modify)
    p.edge(modify, query2)
    p.edge(query2, feedback)
    p.edge(feedback, decision)
    p.edge(modify, stop)
    p.edge(stop, adv, "attack result")


def page_candidates(mxfile):
    p = Page(mxfile, "candidate-construction", "3 - Candidate Construction", 1800, 960)
    p.label("Target-Aware Semantic Candidate Construction", 300, 20, 1200, 55, TITLE)
    colors = ["#E7F5FF", "#FFF7DC", "#EDFAEF"]
    xs = [45, 640, 1235]
    names = ["Observable Target Evidence", "Semantic Action Library", "Candidate Sources"]
    for i, (x, c, name) in enumerate(zip(xs, colors, names), 1):
        p.zone(x, 95, 520, 330, c)
        p.circle(str(i), x + 20, 115, 42, [BLUE, CORAL, GREEN][i - 1])
        p.label(name, x + 75, 108, 410, 50, STAGE)

    add_graph(p, 95, 215, 0.95, target=True)
    rank = p.vertex("Alert Ranking<br><br>Node   Score   Alert<br>●      0.12      –<br>●      0.87      !<br>●      0.43      –", 315, 180, 205, 165, BOX)
    p.vertex("Select Top-K", 345, 360, 145, 38, join_style(BOX, "fillColor=#FFFFFF"))

    actions = []
    for j, (name, color, subtitle) in enumerate([
        ("READ", "#D8EFFF", "Process → File"),
        ("WRITE", "#EBFBEE", "Process → File"),
        ("SENDMSG", "#F3EFFF", "Process → NetFlow"),
    ]):
        actions.append(p.vertex(f"✓  <b>{name}</b><br>{subtitle}", 690, 180 + j * 70, 420, 55, join_style(BOX, f"fillColor={color}")))
    p.label("node type · event type · direction", 740, 385, 320, 30, SMALL)

    local = p.vertex("Local Context<br>two-hop neighborhood", 1285, 180, 205, 105, join_style(BOX, "fillColor=#E7F5FF"))
    benign = p.vertex("Benign Repository<br>graph-wide supplements", 1515, 180, 195, 105, join_style(BOX, "fillColor=#EBFBEE", "shape=cylinder"))
    compat = p.vertex("Available local and global nodes", 1315, 325, 365, 60, join_style(BOX, "fillColor=#FFFFFF"))
    p.edge(local, compat)
    p.edge(benign, compat)

    # bottom native editable workflow
    workflow = [
        ("Target Profile", "target type + local context", "#E7F5FF"),
        ("Valid Action Set", "semantically valid edits", "#FFF7DC"),
        ("Compatibility Gate", "type · relation · direction", "#F3EFFF"),
        ("Candidate Pool", "valid local + benign nodes", "#EDFAEF"),
        ("Action Mask", "selectable target-action pairs", "#FFE8E8"),
    ]
    ids = []
    for i, (head, sub, color) in enumerate(workflow):
        ids.append(p.vertex(f"<b>{head}</b><br>{sub}", 95 + i * 330, 605, 250, 120, join_style(BOX, f"fillColor={color}", "fontSize=16")))
    labels = ["target type", "instantiate", "filter", "mask"]
    for i in range(4):
        p.edge(ids[i], ids[i + 1], labels[i])
    p.edge(rank, ids[0], "ranked alert", style=DASH_EDGE)
    p.edge(actions[1], ids[1], "action semantics", style=DASH_EDGE)
    p.edge(compat, ids[3], "local + global nodes", style=DASH_EDGE)


def page_macro(mxfile):
    p = Page(mxfile, "macro-action-feedback", "4 - Macro-Action Feedback", 1900, 950)
    p.label("Macro-Action Feedback: One Batch, One Query, One Policy Update", 300, 18, 1300, 55, TITLE)
    p.zone(50, 95, 700, 470, "#FFF7DC")
    p.zone(810, 95, 400, 470, "#EAF7FF")
    p.zone(1270, 95, 580, 470, "#F3EFFF")
    p.zone(570, 650, 800, 240, "#EDFAEF")

    p.circle("1", 70, 115, 42, CORAL)
    p.label("Construct the Macro-Action", 125, 108, 560, 50, STAGE)
    state = p.vertex("Current State  Sₜ", 280, 175, 250, 55, BOX)
    strip = p.vertex("L primitive edits applied to the latest graph", 95, 270, 610, 190, join_style(BOX, "fillColor=#FFFFFF"))
    g0 = add_graph(p, 130, 330, 0.7)
    edit = p.vertex("✎<br>edit 1", 330, 330, 100, 80, join_style(BOX, "fillColor=#FFF1BD", "fontSize=20"))
    dots = p.label("···", 455, 345, 70, 40, join_style(TEXT, "fontSize=25", "fontStyle=1"))
    gl = add_graph(p, 545, 330, 0.7)
    no_query = p.vertex("⊘  <b>No detector query inside the batch</b><br>The detector only sees the final modified graph.", 145, 485, 510, 58, join_style(TEXT, "align=left", "fontSize=14"))
    p.edge(state, strip)

    p.circle("2", 830, 115, 42, BLUE)
    p.label("Query Once", 885, 108, 275, 50, STAGE)
    detector = p.vertex("<b>Black-Box Detector</b><br><br>query  →  f(·)", 875, 230, 270, 155, join_style(BOX, "fontSize=19"))
    feedback = p.vertex("✓  <b>Observable Feedback</b><br>scores + alert status", 855, 445, 310, 75, BOX)
    p.edge(detector, feedback)

    p.circle("3", 1290, 115, 42, PURPLE)
    p.label("Compose the Reward", 1345, 108, 440, 50, STAGE)
    score = p.vertex("<b>Score</b><br>reduction", 1315, 190, 230, 80, join_style(BOX, "fillColor=#D8EFFF"))
    threshold = p.vertex("<b>Threshold</b><br>crossing", 1570, 190, 230, 80, join_style(BOX, "fillColor=#D8F7DD"))
    indirect = p.vertex("<b>Indirect</b><br>effect", 1315, 300, 230, 80, join_style(BOX, "fillColor=#FFF0BA"))
    cost = p.vertex("<b>Edit</b><br>cost", 1570, 300, 230, 80, join_style(BOX, "fillColor=#FFE1E1"))
    reward = p.circle("rₜ", 1510, 445, 90, "#FFFFFF", INK)
    p.edge(score, reward, style=DASH_EDGE)
    p.edge(threshold, reward, style=DASH_EDGE)
    p.edge(indirect, reward, style=DASH_EDGE)
    p.edge(cost, reward, style=DASH_EDGE)

    p.circle("4", 590, 670, 42, GREEN)
    p.label("Update the Action-Selection Policy", 645, 663, 600, 50, STAGE)
    gear = p.vertex("⚙", 660, 755, 120, 100, join_style(
        "ellipse", "whiteSpace=wrap", "html=1", "sketch=1", "jiggle=2",
        f"fillColor={GREEN}", f"strokeColor={INK}", "strokeWidth=2", "fontSize=45",
    ))
    policy = p.vertex("<b>Action-Selection Policy</b><br><br>────●────────<br>────────●────", 850, 730, 380, 120, BOX)
    next_state = p.vertex("Next State  Sₜ₊₁  + updated policy", 820, 852, 440, 42, join_style(BOX, "fillColor=#D8EFFF"))
    p.edge(gear, policy)

    # main execution loop
    p.edge(strip, detector, "Gₜ⁽ᴸ⁾")
    p.edge(feedback, score, "feedback")
    p.edge(reward, policy, "use the macro-action outcome", points=[(1580, 620), (1380, 770)])
    p.edge(policy, state, "continue with the latest graph", points=[(450, 840), (250, 650)])


def build():
    metadata = {
        "host": "app.diagrams.net",
        "modified": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "agent": "OpenAI Codex",
        "version": "24.7.17",
        "type": "device",
        "compressed": "false",
    }
    mxfile = ET.Element("mxfile", metadata)
    page_detector(mxfile)
    page_attack(mxfile)
    page_candidates(mxfile)
    page_macro(mxfile)
    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(OUT, encoding="utf-8", xml_declaration=True)
    print(OUT)

    for diagram in mxfile.findall("diagram"):
        filename = INDIVIDUAL_OUTPUTS[diagram.get("id")]
        one_page = ET.Element("mxfile", metadata)
        one_page.append(deepcopy(diagram))
        ET.indent(one_page, space="  ")
        destination = OUT.with_name(filename)
        ET.ElementTree(one_page).write(destination, encoding="utf-8", xml_declaration=True)
        print(destination)


if __name__ == "__main__":
    build()
