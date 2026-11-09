"""Fill report_template.tex with numbers from q1_results.json / q2_results.json.
Discussion paragraphs live in discussion_q1.tex / discussion_q2.tex (written after
inspecting results)."""
import json, re, sys
import numpy as np

q1 = json.load(open("q1_results.json"))
q2 = json.load(open("q2_results.json"))
Ns1 = [100, 500, 1000, 5000, 10000]
S = {int(k): v for k, v in q1["sets"].items()}
P_grid = q1["P_grid"]
conf = np.array(q1["conf_opt"])
counts = conf.sum(axis=0)

def tab(cols, header, rows, caption):
    out = ["\\begin{table}[H]\\centering", "\\begin{tabular}{%s}\\toprule" % cols, header + "\\\\\\midrule"]
    out += [r + "\\\\" for r in rows]
    out += ["\\bottomrule\\end{tabular}", "\\caption{%s}" % caption, "\\end{table}"]
    return "\n".join(out)

conf_tab = tab("lcccc", " & $L=1$ & $L=2$ & $L=3$ & $L=4$",
               [f"$D={i+1}$ & " + " & ".join(str(int(v)) for v in conf[i]) for i in range(4)],
               "Optimal classifier on the test set: number of samples with decision $D=i$ (rows) and true label $L=j$ (columns).")

cv_rows = []
for N in Ns1:
    ce = S[N]["cv_err"]; Pb = S[N]["P_best"]
    cells = [(f"\\textbf{{{ce[str(P)]:.3f}}}" if P == Pb else f"{ce[str(P)]:.3f}") for P in P_grid]
    cv_rows.append(f"{N} & " + " & ".join(cells))
cv_tab = ("\\begin{table}[H]\\centering\\small\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * len(P_grid) +
          "}\\toprule $N$ \\textbackslash{} $P$ & " + " & ".join(map(str, P_grid)) + "\\\\\\midrule\n" +
          "\\\\\n".join(cv_rows) + "\\\\\\bottomrule\\end{tabular}}\n"
          "\\caption{10-fold CV estimates of P(error). Bold marks the selected $P$ (minimum; ties go to the smaller $P$).}\\end{table}")

fin_rows = [f"{N} & {S[N]['P_best']} & {S[N]['n_params']} & {S[N]['cv_err'][str(S[N]['P_best'])]:.3f} & "
            f"{S[N]['pe_train']:.4f} & \\textbf{{{S[N]['pe_test']:.4f}}} & {S[N]['pe_test']-q1['pe_opt']:+.4f}"
            for N in Ns1]
fin_rows.append(f"\\midrule Optimal (true pdf) & -- & -- & -- & -- & \\textbf{{{q1['pe_opt']:.4f}}} & 0")
fin_tab = tab("lcccccc", "$N_{train}$ & $P$ & \\#params & CV P(err.) & train P(err.) & \\textbf{test P(err.)} & gap",
              fin_rows, "Final MLPs, each trained by ML with 10 restarts on its full training set and evaluated on the 100{,}000-sample test set.")

Ks = q2["Ks"]
rate_rows = []
for N in [10, 100, 1000]:
    r = q2["rates"][str(N)]
    sel = np.array(q2["selected"][str(N)])
    best = int(np.argmax(r))
    cells = [(f"\\textbf{{{100*v:.0f}}}" if j == best else f"{100*v:.0f}") for j, v in enumerate(r)]
    rate_rows.append(f"{N} & " + " & ".join(cells) + f" & {sel.mean():.2f} & {int(np.median(sel))}")
rate_tab = ("\\begin{table}[H]\\centering\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * len(Ks) +
            "cc}\\toprule $N$ & " + " & ".join(f"$K{{=}}{K}$" for K in Ks) +
            " & mean $\\hat K$ & median $\\hat K$\\\\\\midrule\n" + "\\\\\n".join(rate_rows) +
            "\\\\\\bottomrule\\end{tabular}}\n\\caption{Selection rate (\\%) of each GMM order over "
            f"{q2['R']} independent repetitions (bold = most frequently selected). True order is $K=4$." "}\\end{table}")

dg = json.load(open("q1_diagnostics.json"))
F = {int(k): v for k, v in dg["final"].items()}; CVd = {int(k): v for k, v in dg["cv"].items()}
conv_rows = [f"{N} & {F[N]['P']} & {'yes' if F[N]['best_converged'] else 'no'} ({F[N]['best_iters']} it.) & "
             f"{F[N]['restarts_converged']}/10 & {CVd[N]['retained_converged']}/{CVd[N]['retained_total']} & "
             f"{CVd[N]['runs_converged']}/{CVd[N]['runs_total']} & [{F[N]['restart_nll_min']:.4f}, {F[N]['restart_nll_max']:.4f}]"
             for N in Ns1]
conv_tab = ("\\begin{center}\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccccc}\\toprule "
            "$N$ & $P$ & final: best restart converged? & final: restarts converged & CV: retained fits converged & "
            "CV: all restarts converged & final: training NLL range over restarts\\\\\\midrule\n" +
            "\\\\\n".join(conv_rows) + "\\\\\\bottomrule\\end{tabular}}\\end{center}")
cm = np.array(dg["conf_mlp10k"])
conf_mlp_tab = tab("lcccc", " & $L=1$ & $L=2$ & $L=3$ & $L=4$",
                   [f"$D={i+1}$ & " + " & ".join(str(int(v)) for v in cm[i]) for i in range(4)],
                   "Test-set confusion counts of the $N=10000$ MLP (rows = decision, columns = true label); compare with the optimal classifier's table above.")
sens = {int(k): v for k, v in json.load(open("q1_cv_sensitivity.json")).items()}
sens_rows = []
for N in (500, 1000):
    for lab, ce, Pb in (("1000", S[N]["cv_err"], S[N]["P_best"]), ("5000", sens[N]["cv_err_5000"], sens[N]["P_best_5000"])):
        cells = [(f"\\textbf{{{ce[str(P)]:.3f}}}" if P == Pb else f"{ce[str(P)]:.3f}") for P in P_grid]
        sens_rows.append(f"{N} & {lab} & " + " & ".join(cells))
sens_tab = ("\\begin{center}\\resizebox{\\linewidth}{!}{\\begin{tabular}{ll" + "c" * len(P_grid) +
            "}\\toprule $N$ & max.\\ iterations & " + " & ".join(f"$P{{=}}{P}$" for P in P_grid) +
            "\\\\\\midrule\n" + "\\\\\n".join(sens_rows) + "\\\\\\bottomrule\\end{tabular}}\\end{center}")
sens_text = (f"The selected sizes are identical: $P={sens[500]['P_best_5000']}$ for $N=500$ and "
             f"$P={sens[1000]['P_best_5000']}$ for $N=1000$, with {sens[500]['converged']}\\% and "
             f"{sens[1000]['converged']}\\% of the retained fits now converged. CV P(error) by $P$ (bold = selected):")
rep = {
    "SENS_TABLE": sens_tab, "SENS_TEXT": sens_text,
    "CONV_TABLE": conv_tab, "CONF_MLP_TABLE": conf_mlp_tab,
    "TEST_COUNTS": ", ".join(str(int(c)) for c in counts),
    "PE_OPT": f"{q1['pe_opt']:.4f}",
    "CONF_TABLE": conf_tab, "CV_TABLE": cv_tab, "FINAL_TABLE": fin_tab,
    "P_GRID": ",".join(map(str, P_grid)),
    "GRAD_ERR": f"${q1['grad_rel_err']:.1e}".replace("e-0", "\\times10^{-") + "}$",
    "GAP_10K": f"{100*(S[10000]['pe_test']-q1['pe_opt']):.2f} percentage points",
    "EIG1": f"{q2['avg_eig'][0]:.2f}", "EIG2": f"{q2['avg_eig'][1]:.2f}",
    "EIG3": f"{q2['avg_eig'][2]:.2f}", "EIG4": f"{q2['avg_eig'][3]:.2f}",
    "DIST12": f"{q2['dist12']:.2f}", "EIGSUM": f"{q2['avg_eig'][0]+q2['avg_eig'][1]:.2f}",
    "R": str(q2["R"]), "LL_TRUE": f"{q2['ll_true']:.3f}",
    "RATE_TABLE": rate_tab,
    "Q1_DISCUSSION": open("discussion_q1.tex").read(),
    "Q2_DISCUSSION": open("discussion_q2.tex").read(),
}
s = open("report_template.tex").read()
for k, v in rep.items():
    s = s.replace(f"<<{k}>>", v)
left = re.findall(r"<<[A-Z0-9_]+>>", s)
assert not left, left
open("report.tex", "w").write(s)
print("report.tex written")
