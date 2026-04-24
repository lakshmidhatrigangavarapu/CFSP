"""
Generate a BLACK-AND-WHITE architecture diagram for the paper:
'Counterfactual Simulation of Extreme Mental Health Scenarios'
Seven-stage pipeline. No colors. IEEE-friendly grayscale.
Output: images/Flow_diagram_v2.png (300 DPI, white bg)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ── Figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 11))
ax.set_xlim(0, 10)
ax.set_ylim(5, 19)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Grayscale palette ───────────────────────────────────────────────
BLACK = '#000000'
DARK  = '#333333'
MED   = '#666666'
LIGHT = '#999999'
VLIGHT= '#CCCCCC'
WHITE = '#FFFFFF'
FILL  = '#F2F2F2'   # very light gray fill for boxes

def draw_box(x, y, w, h, text, fontsize=8, fill=FILL, edge=BLACK,
             text_color=BLACK, lw=1.2):
    """Rounded rectangle centered at (x, y)."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.1',
                         facecolor=fill, edgecolor=edge,
                         linewidth=lw, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', linespacing=1.3, zorder=3)

def draw_stage_label(x, y):
    """Stage number label on the left."""
    pass

def arrow_down(x1, y1, x2, y2, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=lw,
                                shrinkA=2, shrinkB=2), zorder=1)

def arrow_right(x1, y1, x2, y2, lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=lw,
                                shrinkA=1, shrinkB=1), zorder=1)

def side_note(x, y, text, fontsize=6):
    ax.text(x, y, text, ha='left', va='center', fontsize=fontsize,
            color=MED, style='italic', zorder=5)

# ── Layout ──────────────────────────────────────────────────────────
cx = 5.3          # center x (shifted right to leave room for stage labels)
lx = 1.55         # x for stage labels
bw = 6.0          # box width
bh = 0.75         # box height
sp = 0.45         # vertical spacing
ag = 0.06         # arrow gap

# Y positions top-down
Y = {}
y = 18.3
Y['input'] = y;      y -= bh + sp
Y['s1'] = y;         y -= bh + sp + 0.05
Y['s2'] = y;         y -= bh + sp
Y['s3'] = y;         y -= bh + sp
Y['s4'] = y;         y -= bh + sp + 0.08
Y['s5'] = y;         y -= 0.85 + sp
Y['br'] = y;         y -= 0.65 + sp + 0.15
Y['s6'] = y;         y -= 0.85 + sp
Y['tot'] = y;        y -= 0.6 + sp + 0.1
Y['s7'] = y;         y -= bh + sp + 0.05
Y['out'] = y

# ═══════════════════════════════════════════════════════════════════
# INPUT
# ═══════════════════════════════════════════════════════════════════
draw_box(cx, Y['input'], bw, bh,
         'Clinical Note\n(MIMIC-IV Discharge Summary)',
         fontsize=9, fill=WHITE, edge=BLACK, lw=2)
#side_note(cx + bw/2 + 0.15, Y['input'], 'Unstructured text input', fontsize=5.5)
arrow_down(cx, Y['input'] - bh/2 - ag, cx, Y['s1'] + bh/2 + ag)

# ═══════════════════════════════════════════════════════════════════
# STAGE 1
# ═══════════════════════════════════════════════════════════════════
draw_stage_label(lx, Y['s1'])
draw_box(cx, Y['s1'], bw, bh,
         'Clinical Factor Extraction\nLoRA Fine-Tuned Phi-3.5-Mini-Instruct  →  14-Field Structured JSON',
         fontsize=7)
#side_note(cx + bw/2 + 0.15, Y['s1'] + 0.12, 'r=32, α=64', fontsize=5.5)
#side_note(cx + bw/2 + 0.15, Y['s1'] - 0.12, '1.30% params trained', fontsize=5.5)
arrow_down(cx, Y['s1'] - bh/2 - ag, cx, Y['s2'] + bh/2 + ag)

# ═══════════════════════════════════════════════════════════════════
# STAGE 2
# ═══════════════════════════════════════════════════════════════════
draw_stage_label(lx, Y['s2'])
draw_box(cx, Y['s2'], bw, bh,
         'Post-Extraction Normalization\nICD Validation  •  Diagnosis Reconciliation  •  Disease Detection',
         fontsize=7)
arrow_down(cx, Y['s2'] - bh/2 - ag, cx, Y['s3'] + bh/2 + ag)

# ═══════════════════════════════════════════════════════════════════
# STAGE 3
# ═══════════════════════════════════════════════════════════════════
draw_stage_label(lx, Y['s3'])
draw_box(cx, Y['s3'], bw, bh,
         'Context Enrichment\nHousing  •  Trauma  •  Functional Decline  •  Legal  •  Prognosis',
         fontsize=7)
arrow_down(cx, Y['s3'] - bh/2 - ag, cx, Y['s4'] + bh/2 + ag)

# ═══════════════════════════════════════════════════════════════════
# STAGE 4
# ═══════════════════════════════════════════════════════════════════
draw_stage_label(lx, Y['s4'])
draw_box(cx, Y['s4'], bw, bh,
         'Negation-Aware Critical Signal Detection\nSuicide  •  Violence  •  Weapons  —  40-char Lookback Window',
         fontsize=7)
#side_note(cx + bw/2 + 0.15, Y['s4'] + 0.12,
#          '"threatened to jump" → ACTIVE', fontsize=5.5)
#side_note(cx + bw/2 + 0.15, Y['s4'] - 0.12,
#          '"denies SI" → NEGATED', fontsize=5.5)
arrow_down(cx, Y['s4'] - bh/2 - ag, cx, Y['s5'] + 0.85/2 + ag)

# ═══════════════════════════════════════════════════════════════════
# STAGE 5 — Branch Gating
# ═══════════════════════════════════════════════════════════════════
draw_stage_label(lx, Y['s5'])
draw_box(cx, Y['s5'], bw, 0.85,
         'Evidence-Based Branch Gating\nActivate scenario pathways only when\nclinical evidence exists',
         fontsize=7)

# ── Three branches ──────────────────────────────────────────────
by = Y['br']
bw_br = 1.7
bh_br = 0.62
bx_a = cx - 2.05
bx_b = cx
bx_c = cx + 2.05

for bx in [bx_a, bx_b, bx_c]:
    arrow_down(bx, Y['s5'] - 0.85/2 - ag, bx, by + bh_br/2 + ag, lw=1.2)

# Branch A — solid border
draw_box(bx_a, by, bw_br, bh_br,
         'Branch A\nPsychiatric\nDeterioration', fontsize=6.5,
         fill=FILL, edge=BLACK, lw=1.5)
#ax.text(bx_a, by - bh_br/2 - 0.12, '(always active)', ha='center',
#        va='top', fontsize=5.5, color=MED, style='italic')

# Branch B — dashed border (conditional)
box_b = FancyBboxPatch((bx_b - bw_br/2, by - bh_br/2), bw_br, bh_br,
                        boxstyle='round,pad=0.1',
                        facecolor=WHITE, edgecolor=BLACK,
                        linewidth=1.5, linestyle='dashed', zorder=2)
ax.add_patch(box_b)
ax.text(bx_b, by, 'Branch B\nSubstance\nEscalation', ha='center', va='center',
        fontsize=6.5, color=BLACK, fontweight='bold', linespacing=1.3, zorder=3)
#ax.text(bx_b, by - bh_br/2 - 0.12, '(conditional)', ha='center',
#        va='top', fontsize=5.5, color=MED, style='italic')

# Branch C — solid border
draw_box(bx_c, by, bw_br, bh_br,
         'Branch C\nSocial / Env.\nCollapse', fontsize=6.5,
         fill=FILL, edge=BLACK, lw=1.5)
#ax.text(bx_c, by - bh_br/2 - 0.12, '(always active)', ha='center',
#       va='top', fontsize=5.5, color=MED, style='italic')

# Arrows from branches → Stage 6
# Branch B (center) - straight arrow
arrow_down(bx_b, by - bh_br/2 - ag, cx, Y['s6'] + 0.85/2 + ag, lw=1.2)

# Branch A (left) - curved arrow for clarity
from matplotlib.patches import FancyArrowPatch
arrow_a = FancyArrowPatch(
    (bx_a, by - bh_br/2 - ag),
    (cx - bw/2 + 0.4, Y['s6'] + 0.85/2 + ag),
    connectionstyle="arc3,rad=0.15",
    arrowstyle='->', mutation_scale=12,
    color=BLACK, lw=1.3, zorder=1
)
ax.add_patch(arrow_a)

# Branch C (right) - curved arrow for clarity
arrow_c = FancyArrowPatch(
    (bx_c, by - bh_br/2 - ag),
    (cx + bw/2 - 0.4, Y['s6'] + 0.85/2 + ag),
    connectionstyle="arc3,rad=-0.15",
    arrowstyle='->', mutation_scale=12,
    color=BLACK, lw=1.3, zorder=1
)
ax.add_patch(arrow_c)

# ═══════════════════════════════════════════════════════════════════
# STAGE 6 — Tree-of-Thoughts Scenario Generation
# ═══════════════════════════════════════════════════════════════════
draw_stage_label(lx, Y['s6'])
draw_box(cx, Y['s6'], bw, 0.85,
         'Tree-of-Thoughts Scenario Generation\n(per active branch)',
         fontsize=8)

# Three sequential ToT steps
tot_y = Y['tot']
tw = 1.7
th = 0.58
tx_1 = cx - 2.05
tx_2 = cx
tx_3 = cx + 2.05

for tx in [tx_1, tx_2, tx_3]:
    arrow_down(tx, Y['s6'] - 0.85/2 - ag, tx, tot_y + th/2 + ag, lw=1.0)

draw_box(tx_1, tot_y, tw, th, 'Step 1\nTrigger\nIdentification', fontsize=6)
draw_box(tx_2, tot_y, tw, th, 'Step 2\nCausal Chain\nReasoning', fontsize=6)
draw_box(tx_3, tot_y, tw, th, 'Step 3\nScenario\nNarrative', fontsize=6)

# Horizontal arrows between steps
arrow_right(tx_1 + tw/2 + 0.04, tot_y, tx_2 - tw/2 - 0.04, tot_y, lw=1.5)
arrow_right(tx_2 + tw/2 + 0.04, tot_y, tx_3 - tw/2 - 0.04, tot_y, lw=1.5)

# Arrow down to Stage 7
arrow_down(cx, tot_y - th/2 - 0.15, cx, Y['s7'] + bh/2 + ag)

# ═══════════════════════════════════════════════════════════════════
# STAGE 7 — Evidence Attribution
# ═══════════════════════════════════════════════════════════════════
draw_stage_label(lx, Y['s7'])
draw_box(cx, Y['s7'], bw, bh,
         'Evidence Attribution  +  Nexus Factor Detection\nExact & Fuzzy Match (τ=0.45)  •  Cross-Branch Nexus  •  Coverage',
         fontsize=7)
#side_note(cx + bw/2 + 0.15, Y['s7'] + 0.12, 'Maps every claim', fontsize=5.5)
#side_note(cx + bw/2 + 0.15, Y['s7'] - 0.12, 'to source note spans', fontsize=5.5)
arrow_down(cx, Y['s7'] - bh/2 - ag, cx, Y['out'] + 0.8/2 + ag)

# ═══════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════
draw_box(cx, Y['out'], bw, 0.8,
         'Evidence-Attributed Crisis Scenarios\nGrounded Narratives  •  Nexus Factors  •  Coverage Statistics',
         fontsize=8, fill=WHITE, edge=BLACK, lw=2)

# ── Dashed feedback line (Stage 7 traces back to source note) ───
# ax.annotate('', xy=(cx + bw/2 + 0.12, Y['input']),
#             xytext=(cx + bw/2 + 0.12, Y['s7']),
#             arrowprops=dict(arrowstyle='->', color=LIGHT, lw=1.2,
#                             linestyle='dashed'), zorder=0)
# ax.text(cx + bw/2 + 0.05,
#         (Y['input'] + Y['s7'])/2,
#         'traces back to source note',
#         ha='center', va='center', fontsize=5,
#         color=LIGHT, style='italic', rotation=90, zorder=5)

# ── Save ────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'images', 'Flow_diagram_v2.png'
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {output_path}")
