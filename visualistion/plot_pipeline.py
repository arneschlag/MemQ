"""
Visualize the MemQ data pipeline: raw SPARQL → parsed JSON → query graph.
Saves output/pipeline_plot.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import json

plt.rcParams.update({"font.size": 9, "font.family": "monospace"})

# ── Load the example ─────────────────────────────────────────────────────────
with open("output/webqsp_train_graph.json") as f:
    graphs = json.load(f)
g = next(g for g in graphs if g["id"] == "WebQTrn-6")

fig = plt.figure(figsize=(22, 14))
fig.suptitle("MemQ Pipeline: SPARQL → Parse → Query Graph\n"
             f'Example: "{g["question"]}"',
             fontsize=14, fontweight="bold", y=0.98)

# ── Helper ───────────────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, text, color="#E8F0FE", fontsize=8, bold=False):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor="#555", linewidth=1.5,
                         mutation_scale=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, family="monospace",
            wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color="#333", lw=1.5):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1: Pipeline Overview (top strip)
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.02, 0.87, 0.96, 0.10])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1)
ax1.axis("off")

# Boxes
draw_box(ax1, 0.1, 0.15, 2.0, 0.7, "my_webqsp_train.json\n{id, question,\n ori_sparql, BegE}",
         color="#FFE0B2", bold=True)
draw_box(ax1, 3.0, 0.15, 2.2, 0.7, "gen_parse_data.py\nRegex-Parser:\n→ Triples, Filter,\n  EXISTS, AnsE",
         color="#C8E6C9", bold=True)
draw_box(ax1, 6.0, 0.15, 2.0, 0.7, "webqsp_train_parse.json\n{id, where[],\n filter[], exists[],\n AnsE, BegE}",
         color="#FFE0B2")
draw_box(ax1, 8.7, 0.15, 1.2, 0.7, "build_graph\n_train.py",
         color="#C8E6C9", bold=True)

draw_arrow(ax1, 2.1, 0.5, 2.95, 0.5, "#555", 2)
draw_arrow(ax1, 5.2, 0.5, 5.95, 0.5, "#555", 2)
draw_arrow(ax1, 8.0, 0.5, 8.65, 0.5, "#555", 2)

ax1.text(0.1, -0.0, "INPUT", fontsize=8, fontweight="bold", color="#E65100")
ax1.text(3.0, -0.0, "STEP 1", fontsize=8, fontweight="bold", color="#2E7D32")
ax1.text(6.0, -0.0, "INTERMEDIATE", fontsize=8, fontweight="bold", color="#E65100")
ax1.text(8.7, -0.0, "STEP 2", fontsize=8, fontweight="bold", color="#2E7D32")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2: SPARQL decomposition (left side)
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.02, 0.02, 0.45, 0.82])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("Step 1: gen_parse_data.py — Regex Decomposition",
              fontsize=11, fontweight="bold", pad=8)

# SPARQL box
sparql_text = (
    "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    "SELECT DISTINCT ?x\n"
    "WHERE {\n"
    "  FILTER (?x != ns:m.0c2yrf)\n"
    "  FILTER (!isLiteral(?x) OR ...)\n"
    "  ns:m.0c2yrf ns:sports.pro_athlete.teams ?y .\n"
    "  ?y ns:sports.sports_team_roster.team ?x .\n"
    "  FILTER(NOT EXISTS {?y ... from ?sk0} ||\n"
    "         EXISTS {?y ... from ?sk1 .\n"
    "           FILTER(xsd:datetime(?sk1) <= ...)})\n"
    "  FILTER(NOT EXISTS {?y ... to ?sk2} ||\n"
    "         EXISTS {?y ... to ?sk3 .\n"
    "           FILTER(xsd:datetime(?sk3) >= ...)})\n"
    "}"
)
draw_box(ax2, 0.2, 6.5, 9.6, 3.3, sparql_text, color="#F3E5F5", fontsize=6.5)

# Regex labels with arrows pointing into the SPARQL
regexes = [
    (0.5, 9.0, "RE_SELECT", "#BBDEFB"),
    (2.5, 9.0, "RE_FILTER", "#BBDEFB"),
    (4.5, 9.0, "RE_TRIPLE", "#BBDEFB"),
    (6.5, 9.0, "RE_EXISTS_BLOCK", "#BBDEFB"),
]
for x, y, label, color in regexes:
    draw_box(ax2, x-0.4, y-0.25, 1.6, 0.5, label, color=color, fontsize=7, bold=True)

# Result boxes (what gets extracted)
y_base = 5.5
draw_box(ax2, 0.2, y_base, 2.0, 0.7, "AnsE = ?x", color="#E8F5E9", bold=True)

draw_box(ax2, 2.5, y_base-0.8, 3.5, 1.5,
         "where[]:\n"
         "  [ns:m.0c2yrf, teams, ?y]\n"
         "  [?y, team, ?x]",
         color="#E8F5E9", fontsize=6.5)

draw_box(ax2, 6.3, y_base-0.8, 3.5, 1.5,
         "filter[]:\n"
         '  "?x != ns:m.0c2yrf"\n'
         '  "!isLiteral(?x) OR ..."',
         color="#FFF9C4", fontsize=6.5)

draw_box(ax2, 0.2, 3.0, 9.6, 2.3,
         "exists[0] = [?sk0, \"?y ... from ?sk1\", \"(xsd:datetime(?sk1) <= ...)\"]\n"
         "exists[1] = [?sk2, \"?y ... to   ?sk3\", \"(xsd:datetime(?sk3) >= ...)\"]\n\n"
         "Platzhalter (?sk0, ?sk2) = Variable, die NUR in NOT EXISTS vorkommt → Indikator "
         "für optionale Kante\n"
         "Triple-String = die Verbindung, die bei erfülltem EXISTS gilt\n"
         "Filter-Ausdruck = Zusatzbedingung für den EXISTS-Zweig",
         color="#FFEBEE", fontsize=6.5)

# Extraction order arrows
ax2.annotate("1.", xy=(0.1, y_base+0.35), fontsize=14, fontweight="bold", color="#C62828")
ax2.annotate("2.", xy=(2.4, y_base+0.35), fontsize=14, fontweight="bold", color="#C62828")
ax2.annotate("3.", xy=(6.2, y_base+0.35), fontsize=14, fontweight="bold", color="#C62828")
ax2.text(0.1, 3.8,
         "Extraction order: (1) EXISTS blocks, (2) Subqueries, "
         "(3) UNIONs, (4) standalone FILTERs, (5) remaining triples",
         fontsize=7, fontweight="bold", color="#C62828")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3: Graph built by build_graph_train.py (right side)
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_axes([0.50, 0.02, 0.48, 0.82])
ax3.set_xlim(-3, 10)
ax3.set_ylim(-1, 9)
ax3.axis("off")
ax3.set_title("Step 2: build_graph_train.py — Query Graph",
              fontsize=11, fontweight="bold", pad=8)

# ── Graph nodes ──
node_pos = {
    "ns:m.0c2yrf\n(Joakim Noah)": (-1.5, 6.5),
    "?y\n(team roster\n entry)": (0.5, 6.5),
    "?x\n(answer:\n team)": (2.5, 6.5),
    "?sk1\n(from date,\n try edge)": (-0.5, 3.5),
    "?sk3\n(to date,\n try edge)": (1.5, 3.5),
}

node_colors = {
    0: "#BBDEFB",  # start entity
    1: "#FFF9C4",  # intermediate
    2: "#C8E6C9",  # answer variable
    3: "#FFCDD2",  # try edge target
    4: "#FFCDD2",  # try edge target
}

for i, (label, (x, y)) in enumerate(node_pos.items()):
    color = node_colors[i]
    is_answer = "answer" in label
    lw = 3 if is_answer else 1.5
    circle = plt.Circle((x, y), 0.55, facecolor=color, edgecolor="#333",
                        linewidth=lw, zorder=3)
    ax3.add_patch(circle)
    ax3.text(x, y, label, ha="center", va="center", fontsize=6.5,
             fontweight="bold" if is_answer else "normal",
             family="monospace")

# ── Edges ──
edges = [
    ("ns:m.0c2yrf", "?y", "teams", False, 0),
    ("?y", "?x", "team", False, 0),
    ("?y", "?sk1", "from", True, -0.25),
    ("?y", "?sk3", "to", True, 0.25),
]

# Build lookup: short var name → (x, y) position
short_to_pos = {}
for label, pos in node_pos.items():
    short = label.split("\n")[0]
    short_to_pos[short] = pos

for src_label, tgt_label, rel, is_try, y_off in edges:
    src = short_to_pos[src_label]
    tgt = short_to_pos[tgt_label]

    style = "solid" if not is_try else "dashed"
    color = "#555" if not is_try else "#D32F2F"
    lw = 2.5 if not is_try else 1.8

    # Draw arrow from edge of source circle to edge of target circle
    dx = tgt[0] - src[0]
    dy = tgt[1] - src[1]
    dist = (dx**2 + dy**2) ** 0.5
    r = 0.55  # circle radius
    sx, sy = src[0] + dx/dist * r, src[1] + dy/dist * r
    tx, ty = tgt[0] - dx/dist * r, tgt[1] - dy/dist * r

    ax3.annotate("", xy=(tx, ty), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                               linestyle=style, connectionstyle="arc3,rad=0"))

    # Edge label
    mid_x = (sx + tx) / 2
    mid_y = (sy + ty) / 2
    if src_label == "?y" and tgt_label == "?sk1":
        mid_x -= 0.8
        mid_y -= 0.4
    elif src_label == "?y" and tgt_label == "?sk3":
        mid_x += 0.8
        mid_y -= 0.4
    else:
        mid_y += 0.35

    rel_short = rel
    try_str = " ⚡TRY" if is_try else ""
    ax3.text(mid_x, mid_y, f"{rel_short}{try_str}", fontsize=6.5, family="monospace",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.85))

# ── Legend ──
legend_items = [
    (mpatches.Patch(color="#BBDEFB"), "Start-Entity (BegE)"),
    (mpatches.Patch(color="#FFF9C4"), "Zwischenvariable"),
    (mpatches.Patch(color="#C8E6C9"), "Answer-Variable (AnsE)"),
    (mpatches.Patch(color="#FFCDD2"), "Try-Edge-Ziel (aus EXISTS)"),
    (mpatches.Patch(facecolor="white", edgecolor="#555"), "── Hauptkante (is_try=False)"),
    (mpatches.Patch(facecolor="white", edgecolor="#D32F2F"), "- - Try-Kante (is_try=True)"),
]
ax3.legend([p for p, _ in legend_items], [t for _, t in legend_items],
           loc="lower left", fontsize=7, ncol=2, framealpha=0.9)

# ── Filter annotations ──
filter_info = (
    "all_rel['?x'].filter:\n"
    "  • ?x != ns:m.0c2yrf\n"
    "  • !isLiteral(?x) OR langMatches(...)\n\n"
    "all_rel['?sk1'].filter:\n"
    "  • xsd:datetime(?sk1) <= \"2015-08-10\"\n\n"
    "all_rel['?sk3'].filter:\n"
    "  • xsd:datetime(?sk3) >= \"2015-08-10\""
)
draw_box(ax3, -3, -0.5, 5.5, 2.8, filter_info, color="#E3F2FD", fontsize=6.5)

# ── main_path annotation ──
draw_box(ax3, 3.0, -0.5, 6.5, 1.2,
         "main_path: [Joakim Noah → ?y → ?x]\n"
         "→ Der kürzeste Pfad von der Start-Entität zur Answer-Variable\n"
         "  über ausschließlich non-try Kanten",
         color="#C8E6C9", fontsize=7)

# ── Structural annotations ──
ax3.annotate("all_rel dict:\nJeder Zielknoten speichert\n"
             "• father (Vorgänger)\n• relation\n• filter[]\n"
             "• is_try\n• reversed",
             xy=(3.0, 1.5), fontsize=7, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor="#E65100"))

# ── Connect panel 2 → panel 3 visually ──
ax3.annotate("← parse output\n  feeds into\n  graph builder",
             xy=(-2.8, 8.5), fontsize=8, fontweight="bold",
             color="#6A1B9A",
             bbox=dict(boxstyle="round", facecolor="#F3E5F5", edgecolor="#6A1B9A"))

plt.savefig("output/pipeline_plot.png", dpi=180, bbox_inches="tight",
            facecolor="white")
print("Saved output/pipeline_plot.png")
plt.close()
