"""Build the reviewed report from regenerated JSON/figures (ReportLab + mathtext)."""
from pathlib import Path
import os, sys, json, textwrap
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
# Bundled ReportLab is absent from the user's Anaconda; append (do not override) its package directory.
sys.path.append(r'C:\Users\abuda\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
font_dir=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
if (font_dir/'arial.ttf').exists():
    pdfmetrics.registerFont(TTFont('Arial',str(font_dir/'arial.ttf')))
    pdfmetrics.registerFont(TTFont('Arial-Bold',str(font_dir/'arialbd.ttf')))
    pdfmetrics.registerFont(TTFont('Arial-Italic',str(font_dir/'ariali.ttf')))
else:
    from reportlab.pdfbase.pdfmetrics import Font
    for alias, base in [('Arial','Helvetica'),('Arial-Bold','Helvetica-Bold'),('Arial-Italic','Helvetica-Oblique')]:
        pdfmetrics.registerFont(Font(alias,base,'WinAnsiEncoding'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
styles=getSampleStyleSheet()
for k in ['Normal','BodyText']: styles[k].fontName='Arial';styles[k].fontSize=9;styles[k].leading=12
styles.add(ParagraphStyle(name='TitleX',fontName='Arial-Bold',fontSize=23,leading=28,textColor=colors.HexColor('#16324f'),spaceAfter=14))
styles.add(ParagraphStyle(name='HeadX',fontName='Arial-Bold',fontSize=15,leading=19,textColor=colors.HexColor('#16324f'),spaceAfter=10))
styles.add(ParagraphStyle(name='SubX',fontName='Arial-Bold',fontSize=10,leading=13,spaceBefore=7,spaceAfter=5))
styles.add(ParagraphStyle(name='CaptionX',fontName='Arial-Italic',fontSize=8,leading=10,spaceAfter=7))
styles.add(ParagraphStyle(name='CodeX',fontName='Courier',fontSize=6.1,leading=7.5))
styles.add(ParagraphStyle(name='CellX',fontName='Arial',fontSize=7.5,leading=9))
W=492
story=[]
q1=json.loads(Path('out/q1_results.json').read_text());q2=json.loads(Path('out/q2_results.json').read_text());q3=json.loads(Path('out/q3_results.json').read_text())
checks=json.loads(Path('out/verification.json').read_text())
def p(t): story.append(Paragraph(t,styles['BodyText']));story.append(Spacer(1,5))
def h(t): story.append(Paragraph(t,styles['SubX']))
def page(t):
 if story: story.append(PageBreak())
 story.append(Paragraph(t,styles['HeadX']))
def caption(t): story.append(Paragraph(t,styles['CaptionX']))
def img(name,maxh=235):
 path=Path('figs')/name
 iw,ih=PILImage.open(path).size
 scale=min(W/iw,maxh/ih)
 story.append(Image(str(path),width=iw*scale,height=ih*scale));story.append(Spacer(1,5))
def table(rows,widths=None,fs=8):
 cells=[[Paragraph(escape(str(v)),styles['CellX']) for v in row] for row in rows]
 t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#16324f')),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#aaaaaa')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f9')]),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 story.append(t);story.append(Spacer(1,8))
def mat(M,fmt='.4f'):
 return '\n'.join('  '.join(format(v,fmt) for v in row) for row in M)
def mono(t): story.append(Preformatted(t,ParagraphStyle('matrix',fontName='Courier',fontSize=8,leading=10)));story.append(Spacer(1,5))
def eq(s):
 folder=Path('figs/equations');folder.mkdir(exist_ok=True)
 target=folder/f'e{len(list(folder.glob("*.png"))):03}.png'
 fig=plt.figure(figsize=(10,.55));fig.text(.01,.5,'$'+s+'$',fontsize=13,va='center');fig.savefig(target,dpi=180,bbox_inches='tight',pad_inches=.07);plt.close(fig)
 iw,ih=PILImage.open(target).size;scale=min(W/iw,40/ih)
 story.append(Image(str(target),width=iw*scale,height=ih*scale));story.append(Spacer(1,6))
def conf(r,labels,digits=4):
 table([['Decision / true label']+[str(v) for v in labels]]+[[str(labels[i])]+[f'{v:.{digits}f}' for v in row] for i,row in enumerate(r['conf'])],[100]+[(W-100)/len(labels)]*len(labels))
def stats(r):
 rows=[['Class','N','Prior','Rank','lambda','cond(reg)']]
 for c,v in r['per_class'].items(): rows.append([c,v['n'],f"{v['prior']:.4f}",v['rank'],f"{v['lam']:.2e}",f"{v['cond_reg']:.2e}"])
 table(rows,[45,65,75,65,105,137])
# 1
story.append(Paragraph('Machine Learning<br/>Assignment 1',styles['TitleX']))
p('<b>EECE5644 - Fall 2026 | David Abu | Reviewed working draft</b><br/>Regenerated October 16, 2026. Mathematical analysis, experimental results, and code are included in this PDF.')
h('Results at a glance')
table([['Experiment','Error estimate','Meaning'],['Q1: true Gaussian / Naive Bayes / LDA','2.593% / 3.248% / 5.210%','Same 10,000 generated observations'],['Q2: 0-1 / class-3 penalty 10 / 100','17.49% / 27.57% / 43.87%','Higher penalties protect class 3'],['Q3: white wine / HAR','50.327% / 0.301%','Required all-sample training errors']],[200,135,157])
h('External resources and AI assistance')
p('The supplied draft and initial Python implementation were generated with Claude (Anthropic), as identified in the supplied materials. OpenAI Codex reviewed that implementation, corrected ROC integration and validation preprocessing, downloaded original UCI data, reran the experiments, performed numerical checks, and generated this revised report and study guide. The exact Claude model stated in the original draft has not been independently verified. This is an AI-assisted working draft for David to review and understand; it does not assert that David has already verified or independently authored every derivation or line of code.')
p('Software: Python, NumPy, SciPy, pandas, Matplotlib, and ReportLab. Assignment requirements and all numerical parameter values for Q1 come from the instructor\'s Assignment 1. Dataset sources, hashes, and reproducibility instructions appear later in this report and in the accompanying package.')
h('Notation')
p('L denotes the true label, D the classifier decision, and lambda(i,j) the loss incurred by deciding i when the truth is j. Confusion matrices use decision rows and true-label columns; each column sums to one. A density p(x|L=j) is distinct from a posterior class probability P(L=j|x).')
h('Scope of verification')
p(f'All three experiments were rerun locally. {checks["passed"]} numerical checks passed, including agreement with SciPy Gaussian log-densities, an independent rank-based AUC calculation, confusion-matrix totals, and the closed-form LDA direction. Training performance is distinguished from validation performance throughout.')
# 2
page('Question 1 | Model and minimum-risk rule')
p('Generate N=10,000 observations once with NumPy seed 5644. Draw L=0 with probability 0.35 and L=1 with probability 0.65, then draw X from its class Gaussian. The realized counts are N0=3460 and N1=6540. Save X and L in out/q1_data.npz and reuse them in Parts A, B, and C.')
mono('m0 = (-0.5, -0.5, -0.5, -0.5)^T     m1 = (1, 1, 1, 1)^T\nC0 = 0.25 * [[ 2,  -0.5,  0.3, 0],\n             [-0.5, 1,   -0.5, 0],\n             [ 0.3,-0.5,  1,   0],\n             [ 0,   0,    0,   2]]\nC1 = [[ 1,   0.3, -0.2, 0],\n      [ 0.3, 2,    0.3, 0],\n      [-0.2, 0.3,  1,   0],\n      [ 0,   0,    0,   3]]')
h('Part A.1: derive the decision rule')
eq(r'R(i\mid x)=\sum_{j=0}^{1}\lambda_{ij}P(L=j\mid x),\qquad D(x)=\arg\min_i R(i\mid x)')
p('Decide 1 when R(1|x) &lt; R(0|x). Substitute Bayes\' rule and cancel the positive marginal density p(x):')
eq(r'(\lambda_{01}-\lambda_{11})p(x\mid 1)P_1>(\lambda_{10}-\lambda_{00})p(x\mid 0)P_0')
p('When lambda01 &gt; lambda11, divide to obtain the requested likelihood-ratio test. For ordinary error costs, lambda10 &gt;= lambda00 as well, so the threshold is nonnegative.')
eq(r'\frac{p(x\mid 1)}{p(x\mid 0)}>\gamma,\qquad \gamma=\frac{(\lambda_{10}-\lambda_{00})P_0}{(\lambda_{01}-\lambda_{11})P_1}')
p('For arbitrary nonnegative losses, the undivided inequality remains valid: if lambda01-lambda11 is negative, reverse the inequality after division; if it is zero, compare the remaining term directly. Ties have equal conditional risk and may be broken consistently.')
p('Under 0-1 loss, both correct decisions cost 0 and both mistakes cost 1. Therefore gamma*=0.35/0.65=0.5384615, and log(gamma*)=-0.6190392. This is minimum-error classification, also called the MAP class decision.')
h('Numerically stable Gaussian scoring')
eq(r'\log g(x\mid m,C)=-\frac{1}{2}\left[d\log(2\pi)+\log|C|+(x-m)^TC^{-1}(x-m)\right]')
p('Compute s(x)=log g(x|m1,C1)-log g(x|m0,C0), then decide 1 if s(x)&gt;log(gamma). Cholesky factorization supplies the log-determinant and Mahalanobis distance without explicitly inverting C. Unequal covariances make the decision boundary quadratic.')
# 3
page('Question 1 | ROC and minimum-error operating points')
p('For each threshold, estimate TPR as the fraction of true class-1 samples decided 1 and FPR as the fraction of true class-0 samples decided 1. Sort scores, process all equal scores together, and use the strict rule score &gt; threshold. The initial threshold -infinity gives (FPR,TPR)=(1,1); the maximum observed score gives (0,0).')
eq(r'\widehat P_e(\gamma)=0.35\,\widehat{\mathrm{FPR}}(\gamma)+0.65\,[1-\widehat{\mathrm{TPR}}(\gamma)]')
rows=[['Classifier','Threshold','TPR','FPR','Error','AUC']]
for k,nm in [('A','True pdf'),('B','Naive Bayes'),('C','Fisher LDA')]:
 v=q1[k];th=f"gamma={v['gamma_emp']:.4f}" if k!='C' else f"tau={v['thr_mid']:.5f}"
 rows.append([nm,th,f"{v['tpr']:.4f}",f"{v['fpr']:.4f}",f"{v['perr_min']:.5f}",f"{v['auc']:.5f}"])
table(rows,[85,120,65,65,80,77])
img('q1a_roc.png',240)
caption('True-distribution ROC. Red circle: empirical minimum; black cross: prior-derived threshold.')
p(f"At gamma*=0.53846 the estimated error is {q1['A']['theory_perr']:.6f}; the empirical minimum is {q1['A']['perr_min']:.6f} at gamma={q1['A']['gamma_emp']:.6f}. The empirical minimizer lies in [{q1['A']['gamma_emp_lo']:.6f}, {q1['A']['gamma_emp_hi']:.6f}); every threshold in this interval yields the same decisions.")
p(f"An independent 2,000,000-observation Monte Carlo run gives Bayes-rule error {q1['perr_bayes_mc']:.7f}. Its binomial standard error is {q1['perr_bayes_mc_se']:.7f} (approximate 95% Monte Carlo interval: estimate +/- {1.96*q1['perr_bayes_mc_se']:.7f}). This is an estimate, not an exact analytic Bayes error.")
p('The threshold selected on 10K observations adapts to sampling noise and its reported minimum is optimistically selected on those observations. It does not disprove the population optimum gamma*. In Parts A and B, errors use the specified priors rather than the slightly different realized class proportions.')
# 4
page('Question 1 | Naive Bayes and Fisher LDA')
h('Part B: incorrect independence model')
p('Keep the data unchanged, but score it with diagonal covariances: C0_NB=diag(0.5,0.25,0.25,0.5) and C1_NB=diag(1,2,1,3). These Gaussian models retain true means and variances but discard correlations. Repeat the same ROC sweep and prior-weighted error calculation.')
p(f"The best empirical error rises from {q1['A']['perr_min']:.5f} to {q1['B']['perr_min']:.5f}, a {100*(q1['B']['perr_min']/q1['A']['perr_min']-1):.1f}% relative increase. AUC falls from {q1['A']['auc']:.5f} to {q1['B']['auc']:.5f}. At gamma*=0.53846 the NB error is {q1['B']['theory_perr']:.5f}; that threshold is optimal for the assumed model, not guaranteed optimal for the actual distribution.")
h('Part C: estimate moments and learn one projection')
eq(r'\hat m_j=\frac{1}{N_j}\sum_{n:L_n=j}x_n,\qquad \hat C_j=\frac{1}{N_j}\sum_{n:L_n=j}(x_n-\hat m_j)(x_n-\hat m_j)^T')
p('Use equal class weights in the scatter matrices as instructed. Choose SB=(m1_hat-m0_hat)(m1_hat-m0_hat)^T and SW=C0_hat+C1_hat. Alternative common scalar normalizations change eigenvalues, but not the maximizing direction.')
eq(r'J(w)=\frac{w^TS_Bw}{w^TS_Ww},\qquad S_Bw=\lambda S_Ww,\qquad w\propto S_W^{-1}(\hat m_1-\hat m_0)')
p('Choose the generalized eigenvector for the largest eigenvalue, normalize it to unit length, and orient its sign so that class 1 has the larger projected mean.')
mono('m0_hat: '+str([round(x,5) for x in q1['lda']['mu0_hat']])+'\nm1_hat: '+str([round(x,5) for x in q1['lda']['mu1_hat']])+'\nw_LDA:  '+str([round(x,5) for x in q1['lda']['w']]))
h('Estimated within-class covariance matrices')
mono('C0_hat =\n'+mat(q1['lda']['S0_hat'])+'\n\nC1_hat =\n'+mat(q1['lda']['S1_hat']))
p(f"The largest generalized eigenvalue is {q1['lda']['eig']:.6f} under this normalization. The eigenvector agrees with the normalized closed-form direction. Classify using w^T x &gt; tau, sweeping all projected scores; select tau={q1['C']['thr_mid']:.7f}. For LDA use the sample-count error as requested: errors/N={q1['C']['count_check']:.4f}.")
# 5
page('Question 1 | Comparison and interpretation')
img('q1b_roc.png',190)
img('q1c_roc.png',190)
img('q1_compare.png',190)
caption('Top: Naive Bayes ROC. Middle: full Fisher LDA ROC. Bottom: comparison with empirical minimum-error operating points marked.')
page('Question 1 | Why the thresholds and classifiers differ')
img('q1_perr_vs_thr.png',210)
p('The true-pdf rule performs best because it uses the correct joint density and priors. Naive Bayes loses information about correlations but retains different class variances, so its boundary remains quadratic. Fisher LDA uses one linear projection and a single threshold, yielding a hyperplane; it cannot reproduce the curved Bayes boundary in this unequal-covariance example.')
p('Fisher\'s criterion optimizes a separation-to-spread ratio. It is not generally an error-minimization objective for arbitrary distributions. Equal-covariance Gaussian classes give a Bayes discriminant in the Fisher direction with the appropriate threshold. Equal weights are used here to match the question; priors still affect operating-point selection. Estimated ROCs can cross locally through finite-sample variation, so the observed ordering should not be claimed to hold at every possible threshold.')
# 6
page('Question 2 | Construct a three-class mixture')
p('Choose four 3D Gaussian components with means at the vertices of a regular tetrahedron of edge 2.5. Their average standard deviation, defined as the average of sqrt(trace(Ck)/3), is %.6f. Thus every pairwise mean distance is about %.3f times this scale, creating overlap.'%(q2['avg_std'],2.5/q2['avg_std']))
mono('Component means:\n'+mat(q2['means']))
for i,C in enumerate(q2['covs']): mono(f'C{i+1} =\n'+mat(C,'.2f'))
p('The class priors are (0.3,0.3,0.4). Class 1 uses component 1, class 2 uses component 2, and class 3 is an equal mixture of components 3 and 4:')
eq(r'p(x\mid 3)=\frac{1}{2}g(x\mid\mu_3,\Sigma_3)+\frac{1}{2}g(x\mid\mu_4,\Sigma_4)')
p('For N=10,000, seed 2026 gives class counts '+str(q2['class_counts'])+'. Draw a label first; for class 3 select either component with probability 0.5; draw its Gaussian observation. Save data and component identifiers in out/q2_data.npz.')
h('Part A: minimum probability of error')
eq(r'D(x)=\arg\max_j P(L=j\mid x)=\arg\max_j\left[\log p(x\mid j)+\log P_j\right]')
p('For class 3, evaluate log(0.5*g3+0.5*g4) using log-sum-exp. The mixture density is not a Gaussian centered at the average component mean. Under 0-1 loss, conditional risk is 1-P(L=i|x), so maximizing the posterior minimizes risk.')
# 7
page('Question 2 | MAP results and data visualization')
conf(q2['0-1'],[1,2,3])
p('Each entry estimates P(D=i|L=j) by dividing the corresponding decision/true-label count by the number of observations in true class j. Columns sum to one.')
mono('Decision/true-label count matrix:\n'+mat(q2['0-1']['counts'],'d'))
p(f"MAP error = {q2['0-1']['perr']:.4f}; this equals the observed mean 0-1 loss. The average conditional minimum risk is {q2['0-1']['exp_risk_cond']:.4f}; its difference from realized loss reflects finite sampling.")
img('q2_scatter_01.png',310)
caption('Two views of the same 10K observations. True class is encoded by marker shape (1: dot, 2: triangle, 3: square); green is correct and red is incorrect.')
p('Overlapping Gaussian components make some errors unavoidable even with the true model. Component identity and class identity differ: both components 3 and 4 count as class 3, so confusing those components is not a classification error.')
# 8
page('Question 2 | Unequal costs and expected risk')
mono('Lambda_10  = [[0,1,10], [1,0,10], [1,1,0]]\nLambda_100 = [[0,1,100],[1,0,100],[1,1,0]]')
p('Columns represent true labels. A class-3 observation assigned label 1 or 2 costs w=10 or w=100; mistakes on true classes 1 or 2 cost 1. Define pj=P(L=j|x).')
eq(r'R(1\mid x)=p_2+wp_3,\quad R(2\mid x)=p_1+wp_3,\quad R(3\mid x)=p_1+p_2')
eq(r'D(x)=\arg\min_i\sum_j\lambda_{ij}P(L=j\mid x)')
p('Class 3 is preferred when 1-p3 &lt; min(p1,p2)+w*p3, equivalently p3 &gt; [1-min(p1,p2)]/(w+1), apart from ties. Increasing w expands the region assigned class 3. This is a posterior-dependent condition, not a fixed standalone threshold independent of p1 and p2.')
rows=[['Loss','Observed risk','Conditional-risk average','Error','Class-3 recall']]
for key,nm in [('0-1','0-1'),('10','Lambda10'),('100','Lambda100')]:
 r=q2[key];rows.append([nm,f"{r['risk']:.4f}",f"{r['exp_risk_cond']:.4f}",f"{r['perr']:.4f}",f"{r['conf'][2][2]:.4f}"])
table(rows,[85,100,130,77,100])
p('Observed risk is the sample mean of lambda(Dn,Ln). The conditional-risk average is the sample mean of min_i R(i|xn), using the known generating distribution. These are two Monte Carlo estimators of the same population risk; they need not match exactly on one dataset.')
img('q2_confusions.png',190)
caption('Decision rows and true-label columns. The class-3 diagonal increases as its missed-detection penalty grows.')
h('Tradeoff')
p('As the penalty increases, class-3 recall rises from 0.8499 to 0.9827 to 0.9988, while overall error rises from 0.1749 to 0.2757 to 0.4387. More observations from classes 1 and 2 are deliberately assigned class 3 to avoid expensive class-3 misses. Minimum expected risk is not the same objective as minimum error when losses differ.')
# 9
page('Question 2 | How changing the penalty changes decisions')
img('q2_scatter_10.png',185)
img('q2_scatter_100.png',185)
img('q2_sweep.png',170)
caption('Top: penalty 10. Middle: penalty 100. Bottom: behavior across increasing penalties. All experiments reuse the same data.')
p('For penalty 100, only five true class-3 observations are missed; they contribute 500/10000=0.05 to observed risk. The 4335 misclassified class-1/2 observations contribute 0.4335, and the remaining five class-1/2 cross-errors contribute 0.0005, giving 0.4885? The exact confusion-count calculation below is authoritative.')
# Replace arithmetic paragraph with exact count-based decomposition rather than hand-copying.
story.pop();story.pop()
c=q2['100']['counts'];miss=c[0][2]+c[1][2];cross=c[0][1]+c[1][0];into3=c[2][0]+c[2][1]
p(f'For penalty 100, {miss} true class-3 observations are missed, contributing {miss*100/10000:.4f} to observed risk. The {into3} class-1/2 observations assigned class 3 contribute {into3/10000:.4f}; the {cross} class-1/2 cross-errors contribute {cross/10000:.4f}. Total: {q2["100"]["risk"]:.4f}. As w tends to infinity, the limiting rule assigns class 3 almost everywhere, with population risk P(L=1)+P(L=2)=0.6. The observed finite-sample limit is the realized fraction of classes 1 and 2.')
# 10
page('Question 3 | Estimation, preprocessing, and regularization')
p('Fit one full-covariance Gaussian per observed class. Samples are treated as identically distributed independent observations for estimation, although HAR windows and repeated subjects may violate independence in practice. Class means, covariances, and priors use all available samples for the required final training evaluation.')
eq(r'\hat P_j=N_j/N,\quad \hat m_j=N_j^{-1}\sum_{n:L_n=j}x_n,\quad \hat C_j=N_j^{-1}\sum_{n:L_n=j}(x_n-\hat m_j)(x_n-\hat m_j)^T')
p('Standardize features using their mean and standard deviation. For all-sample evaluation, fit this transformation on all samples. For wine cross-validation, fit it separately on each training fold; for HAR validation, fit it only on the official training portion. Apply the fitted transformation unchanged to the corresponding validation portion.')
p('Without regularization, invertible affine preprocessing preserves Gaussian discriminant decisions because the same Jacobian factor cancels between classes. Adding lambda*I is not affine invariant; standardization gives comparable units before adding isotropic variance.')
eq(r'C_{j,\mathrm{reg}}=\hat C_j+\lambda_j I,\qquad \lambda_j=\alpha\frac{\mathrm{tr}(\hat C_j)}{\mathrm{rank}(\hat C_j)}')
p('A positive lambda shifts every covariance eigenvalue upward; for a singular positive semidefinite estimate, the smallest eigenvalue becomes lambda (not strictly greater). Numerical rank uses an eigenvalue tolerance proportional to matrix size and machine precision. This prevents singular Gaussian calculations and controls unstable low-variance directions.')
eq(r'D(x)=\arg\max_j\left[-\frac{1}{2}(x-\hat m_j)^TC_{j,\mathrm{reg}}^{-1}(x-\hat m_j)-\frac{1}{2}\log|C_{j,\mathrm{reg}}|+\log\hat P_j\right]')
h('Regularization selection')
p('Wine uses stratified five-fold CV and selects alpha=0.001. HAR uses the official 7352/2947 split as validation and selects alpha=0.0001. Because that split selects alpha, its reported error is a tuning/validation result, not an untouched final-test estimate. After selection, refit on all samples as required by the assignment.')
rows=[['alpha','Wine training','Wine CV','HAR training','HAR validation']]
for a,b in zip(q3['wine_sweep'],q3['har_sweep']): rows.append([f"{a['alpha']:g}",f"{a['train']:.4f}",f"{a['cv']:.4f}",f"{b['train']:.4f}",f"{b['test']:.4f}"])
table(rows,[65,105,105,105,112])
p('Neither training nor validation error is guaranteed to vary monotonically with regularization. The observed HAR training sweep is nonmonotonic, so smaller alpha cannot simply be described as always reducing training error.')
# 11
page('Question 3 | White-wine results')
p('The UCI white-wine dataset has 4898 observations, 11 features, and seven observed quality labels: 3 through 9. Scores 0-2 and 10 have no observations and are not fitted classes. Use the white dataset rather than combining red and white wine, matching the stated sample count.')
stats(q3['wine'])
p(f"At alpha={q3['wine']['alpha']}, the required all-sample error is {q3['wine']['perr']:.7f} ({100*q3['wine']['perr']:.3f}%). Always predicting quality 6 gives error {q3['wine']['majority_err']:.4f}; the fitted model improves exact-grade classification modestly. Within one quality grade, accuracy is {q3['wine']['within1']:.4f}.")
img('q3_wine_conf.png',280)
caption('Every cell reports P(D=i|L=j) and its raw count. Most confusion occurs among neighboring quality labels.')
p('Quality 9 has only five observations in 11 dimensions; its sample covariance rank is at most four. Its perfect training classification is therefore weak evidence of performance on new quality-9 wines. Class imbalance and overlapping chemical measurements make exact quality prediction difficult under a single Gaussian per class.')
# 12
page('Question 3 | Human activity recognition results')
p('Combine the original 7352 training and 2947 test observations for the required final fit and training evaluation: N=10299, d=561, six activity labels. The official split is retained only for the separate alpha-validation experiment.')
stats(q3['har'])
p(f"At alpha={q3['har']['alpha']}, all-sample error is {q3['har']['perr']:.7f}, or 31/10299 observations. The selected official-split validation error is {min(v['test'] for v in q3['har_sweep']):.7f} (8.18%). These answer different questions and must not be equated.")
img('q3_har_conf.png',280)
caption('All-sample confusion matrix. Errors concentrate in sitting versus standing and walking downstairs versus upstairs.')
p('Estimated covariance ranks are approximately 468-469 rather than 561, with very large raw condition numbers. This establishes numerical dependence or near-dependence among features. It does not, by itself, identify which engineered features cause it: nonlinear deterministic relationships alone do not imply linear covariance rank deficiency. Regularization permits stable density evaluation in all 561 coordinates.')
p('High-dimensional class-specific covariance models are flexible, and the same observations are used for fitting and scoring. The 0.301% training error therefore cannot establish near-perfect accuracy on new subjects. The much larger validation error illustrates this limitation.')
# 13
page('Question 3 | Wine projections and Gaussian assumptions')
img('q3_wine_pca.png',190)
img('q3_wine_pairs.png',165)
img('q3_wine_hist.png',190)
caption('Top: first three PCA coordinates. Middle: selected raw feature pairs. Bottom: quality-6 marginal distributions.')
p('PCA is computed on centered standardized features for visualization; the classifier still uses all 11 features. The first three components explain %.2f%%, %.2f%%, and %.2f%% of total standardized variance.'%tuple(100*v for v in q3['wine_pca_explained']))
p('Strong overlap between quality classes is visible in the projections. Several within-class marginals have skewness and long tails rather than symmetric bell shapes. A multivariate Gaussian requires Gaussian marginals, so these patterns challenge that assumption. Conversely, approximately Gaussian marginals alone would not establish a Gaussian joint distribution. Model mismatch can contribute to neighboring-quality confusion; these plots do not isolate its contribution from intrinsic overlap and label uncertainty.')
# 14
page('Question 3 | HAR projections and diagnostic limits')
img('q3_har_pca.png',180)
img('q3_har_pairs.png',155)
img('q3_qq.png',150)
caption('PCA and raw-feature views, followed by descriptive squared-Mahalanobis Q-Q plots for selected classes.')
p('For HAR, PC1 explains %.2f%% and PCs 1-3 together explain %.2f%% of variance. Low-dimensional plots show structure and overlap but cannot determine whether all 561-dimensional classes are separable.'%(100*q3['har_pca_explained'][0],100*sum(q3['har_pca_explained'])))
p('With known, full-rank Gaussian parameters in d dimensions, squared Mahalanobis distance follows chi-square(d). Here parameters are fitted on the same samples and covariances are regularized, with substantial rank deficiency in HAR. The nominal chi-square reference is consequently not a calibrated goodness-of-fit test. Deviations are descriptive and can reflect estimation, shrinkage, effective dimension, and non-Gaussianity. They should not be presented as standalone proof of non-Gaussian classes.')
# 15
page('Reproducibility, source records, and review notes')
h('How to rerun')
p('Use Python 3.10+ with NumPy 2+, SciPy, pandas, Matplotlib, and Pillow. From the extracted project folder run: python run_all.py. The runner downloads original UCI archives if needed, creates output folders, runs q1.py, q2.py and q3.py, and checks results. Raw datasets are not bundled in the compact code package; the download script recreates them. PDF generation additionally needs ReportLab and suitable fonts.')
h('Source records')
p('Instructor: EECE5644 Fall 2026, Assignment 1, supplied PDF. Dataset providers:')
p('<link href="https://archive.ics.uci.edu/dataset/186/wine+quality" color="#164b78">UCI Wine Quality, dataset 186</link><br/><link href="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones" color="#164b78">UCI Human Activity Recognition Using Smartphones, dataset 240</link>')
for src in json.loads(Path('data/sources.json').read_text()):
 p('<b>'+src['dataset']+'</b>: '+str(src['bytes'])+' archive bytes. SHA-256:')
 mono('\n'.join(textwrap.wrap(src['sha256'],64)))
p('Archives were downloaded from the original UCI static endpoints on October 16, 2026. The earlier draft\'s mirror identity claim is not reused; matching dimensions alone does not establish byte identity. download_data.py and data/sources.json record the exact sources used for this rerun.')
h('Changes relative to the supplied draft')
for t in ['Replaced the machine-specific /home/claude/data path with a project-relative path and provided a one-command runner and data downloader.','Corrected AUC integration to preserve the order of vertical ROC segments; checked it against an independent rank statistic. The three AUCs round to 0.99589, 0.99420, and 0.98536.','Fitted validation preprocessing only on the training portion. Kept the required all-sample estimates separate from validation estimates.','Qualified the HAR alpha-selection split as validation; removed unsupported claims about exact model provenance, covariance dependence causes, monotonic training-error behavior, and calibrated Q-Q testing.','Saved ROC thresholds/rates, data, numerical summaries, and verification records. Added the following executable source appendix.']:
 p('- '+t)
p('The unchanged uploaded PDF and ZIP remain the originals. This report is a revised working draft; David should review the reasoning, results, and AI acknowledgment before submitting. No submission or upload to Canvas has been performed.')
# Source appendix, include all main experiment code and downloader/runner/verification.
for filename in ['common.py','q1.py','q2.py','q3.py','download_data.py','run_all.py','verify_results.py']:
 path=Path(filename)
 if not path.exists(): continue
 raw=path.read_text(encoding='utf-8-sig')
 lines=[]
 for n,line in enumerate(raw.splitlines(),1):
  wrapped=textwrap.wrap(line.expandtabs(4),width=107,replace_whitespace=False,drop_whitespace=False) or ['']
  lines.append(f'{n:3d} '+wrapped[0]);lines.extend('    '+v for v in wrapped[1:])
 for start in range(0,len(lines),77):
  page('Source appendix | '+filename+(' (continued)' if start else ''))
  story.append(Preformatted('\n'.join(lines[start:start+77]),styles['CodeX']))
def footer(canvas,doc):
 canvas.saveState();canvas.setStrokeColor(colors.HexColor('#cbd5df'));canvas.line(60,43,552,43)
 canvas.setFont('Arial',8);canvas.setFillColor(colors.HexColor('#586575'));canvas.drawString(60,30,'EECE5644 | Assignment 1 | Reviewed working draft');canvas.drawRightString(552,30,str(doc.page));canvas.restoreState()
output=ROOT/'EECE5644_Assignment1_Reviewed.pdf'
doc=SimpleDocTemplate(str(output),pagesize=(612,792),rightMargin=60,leftMargin=60,topMargin=48,bottomMargin=55,title='EECE5644 Assignment 1 - Reviewed working draft',author='David Abu; AI assistance disclosed')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(output)
