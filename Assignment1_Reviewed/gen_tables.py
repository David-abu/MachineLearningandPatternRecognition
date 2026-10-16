"""Write LaTeX table fragments from the saved JSON results (avoids hand-copying numbers)."""
import json
import numpy as np

q1 = json.load(open("out/q1_results.json"))
q2 = json.load(open("out/q2_results.json"))
q3 = json.load(open("out/q3_results.json"))


def conf_table(M, rows, cols, fmt="{:.4f}", counts=None):
    M = np.array(M)
    s = "\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n"
    s += " & " + " & ".join(cols) + " \\\\\n\\midrule\n"
    for i, r in enumerate(rows):
        cells = []
        for j in range(len(cols)):
            v = fmt.format(M[i, j])
            if i == j:
                v = "\\textbf{" + v + "}"
            if counts is not None:
                v += f" ({counts[i][j]})"
            cells.append(v)
        s += r + " & " + " & ".join(cells) + " \\\\\n"
    return s + "\\bottomrule\n\\end{tabular}\n"


# ---- Q1 summary
A, B, C = q1["A"], q1["B"], q1["C"]
s = "\\begin{tabular}{lccccc}\n\\toprule\n"
s += "Classifier & threshold at min $P(e)$ & $P(D{=}1|L{=}1)$ & $P(D{=}1|L{=}0)$ & $\\hat P(\\text{error})$ & AUC \\\\\n\\midrule\n"
s += f"A: true pdf & $\\hat\\gamma = {A['gamma_emp']:.3f}$ ($\\ln\\hat\\gamma={A['thr_mid']:.3f}$) & {A['tpr']:.4f} & {A['fpr']:.4f} & {A['perr_min']:.4f} & {A['auc']:.4f} \\\\\n"
s += f"\\quad at theory $\\gamma^*={q1['gamma_theory']:.4f}$ & $\\ln\\gamma^*={np.log(q1['gamma_theory']):.3f}$ & {A['theory_tpr']:.4f} & {A['theory_fpr']:.4f} & {A['theory_perr']:.4f} & \\\\\n"
s += f"B: naive Bayes & $\\hat\\gamma = {B['gamma_emp']:.3f}$ ($\\ln\\hat\\gamma={B['thr_mid']:.3f}$) & {B['tpr']:.4f} & {B['fpr']:.4f} & {B['perr_min']:.4f} & {B['auc']:.4f} \\\\\n"
s += f"\\quad at $\\gamma^*={q1['gamma_theory']:.4f}$ & $\\ln\\gamma^*={np.log(q1['gamma_theory']):.3f}$ & {B['theory_tpr']:.4f} & {B['theory_fpr']:.4f} & {B['theory_perr']:.4f} & \\\\\n"
s += f"C: Fisher LDA & $\\hat\\tau = {C['thr_mid']:.4f}$ & {C['tpr']:.4f} & {C['fpr']:.4f} & {C['perr_min']:.4f} & {C['auc']:.4f} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
open("out/t_q1.tex", "w").write(s)

# ---- Q2 confusion matrices
for key in ("0-1", "10", "100"):
    t = conf_table(q2[key]["conf"], ["$D=1$", "$D=2$", "$D=3$"], ["$L=1$", "$L=2$", "$L=3$"],
                   counts=q2[key]["counts"])
    open(f"out/t_q2_{key.replace('-', '')}.tex", "w").write(t)
s = "\\begin{tabular}{lccccc}\n\\toprule\nLoss & $\\hat{R}_{\\min}=\\frac1N\\sum_n\\lambda_{D_nL_n}$ & $\\frac1N\\sum_n \\min_i R(D{=}i|x_n)$ & $\\hat P(\\text{error})$ & \\#$\\{D=1,2,3\\}$ \\\\\n\\midrule\n"
for key, nm in (("0-1", "0--1"), ("10", "$\\Lambda_{10}$"), ("100", "$\\Lambda_{100}$")):
    r = q2[key]
    s += f"{nm} & {r['risk']:.4f} & {r['exp_risk_cond']:.4f} & {r['perr']:.4f} & {r['dec_counts'][0]}, {r['dec_counts'][1]}, {r['dec_counts'][2]} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
open("out/t_q2_sum.tex", "w").write(s)

# ---- Q3
w = q3["wine"]
cls = w["classes"]
open("out/t_q3_wine.tex", "w").write(conf_table(w["conf"], [f"$D={c}$" for c in cls], [f"$L={c}$" for c in cls], "{:.3f}"))
h = q3["har"]
short = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]
open("out/t_q3_har.tex", "w").write(conf_table(h["conf"], [f"$D$={n}" for n in short], [f"$L$={n}" for n in short], "{:.4f}"))

s = "\\begin{tabular}{lrrrrrr}\n\\toprule\nClass & $N_j$ & $\\hat P(L{=}j)$ & rank$(\\hat C_j)$ & $\\lambda_j$ & cond$(\\hat C_j)$ & cond$(\\hat C_j+\\lambda_j I)$ \\\\\n\\midrule\n"
for c in cls:
    p = w["per_class"][str(c)]
    s += f"{c} & {p['n']} & {p['prior']:.4f} & {p['rank']} & {p['lam']:.2e} & {p['cond_raw']:.2e} & {p['cond_reg']:.2e} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
open("out/t_q3_wine_cls.tex", "w").write(s)
s = "\\begin{tabular}{lrrrrrr}\n\\toprule\nClass & $N_j$ & $\\hat P(L{=}j)$ & rank$(\\hat C_j)$ & $\\lambda_j$ & cond$(\\hat C_j)$ & cond$(\\hat C_j+\\lambda_j I)$ \\\\\n\\midrule\n"
for i, c in enumerate(sorted(int(k) for k in h["per_class"])):
    p = h["per_class"][str(c)]
    s += f"{c} {short[i]} & {p['n']} & {p['prior']:.4f} & {p['rank']} & {p['lam']:.2e} & {p['cond_raw']:.1e} & {p['cond_reg']:.2e} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
open("out/t_q3_har_cls.tex", "w").write(s)

s = "\\begin{tabular}{lcccc}\n\\toprule\n$\\alpha$ & Wine: training & Wine: 5-fold CV & HAR: training & HAR: validation \\\\\n\\midrule\n"
for a, b in zip(q3["wine_sweep"], q3["har_sweep"]):
    s += f"{a['alpha']:g} & {a['train']:.4f} & {a['cv']:.4f} & {b['train']:.4f} & {b['test']:.4f} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
open("out/t_q3_sweep.tex", "w").write(s)
print("ok")

from pathlib import Path
sources=json.load(open('data/sources.json'))
lines=[]
for src in sources:
    h=src['sha256']
    lines.append(src['dataset']+r': \\ '+h[:32]+r'\\ '+h[32:]+r'\\[3pt]')
Path('out/source_hashes.tex').write_text('\n'.join(lines))
