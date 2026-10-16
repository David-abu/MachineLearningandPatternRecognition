"""Independent checks of numerical helpers and saved assignment results."""
import json, ast
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal, rankdata
from common import gauss_logpdf, roc_sweep, auc
root=Path(__file__).resolve().parent
checks=[]
def check(name,condition):
 if not condition: raise AssertionError(name)
 checks.append(name)
rng=np.random.default_rng(4)
A=rng.normal(size=(4,4));C=A@A.T+np.eye(4);m=rng.normal(size=4);X=rng.normal(size=(30,4))
check('Gaussian log-density agrees with scipy.stats',np.allclose(gauss_logpdf(X,m,C),multivariate_normal.logpdf(X,m,C),atol=1e-11))
scores=np.array([0.,0.,1.,2.,2.,3.]);labels=np.array([0,1,0,1,0,1])
t,f,tpr=roc_sweep(scores,labels)
check('Tied-score ROC matches direct decisions at every threshold',all(np.isclose(f[i],np.mean((scores>v)[labels==0])) and np.isclose(tpr[i],np.mean((scores>v)[labels==1])) for i,v in enumerate(t)))
ranks=rankdata(scores);u=(ranks[labels==1].sum()-3*4/2)/9
check('ROC area agrees with independent rank statistic',np.isclose(auc(f,tpr),u))
r=json.loads((root/'out/q1_results.json').read_text())
for key in ['A','B','C']:
 c=np.load(root/f'out/q1_roc_{key}.npz');check(f'Q1 {key}: ROC endpoints and monotonicity',c['fpr'][0]==1 and c['tpr'][0]==1 and c['fpr'][-1]==0 and c['tpr'][-1]==0 and np.all(np.diff(c['fpr'])<=0) and np.all(np.diff(c['tpr'])<=0))
 check(f'Q1 {key}: AUC agrees with saved ROC',np.isclose(auc(c['fpr'],c['tpr']),r[key]['auc']))
check('LDA generalized eigenvector agrees with closed form',np.allclose(r['lda']['w'],r['lda']['w_closed']))
r2=json.loads((root/'out/q2_results.json').read_text())
for key,w in [('0-1',1),('10',10),('100',100)]:
 v=r2[key];M=np.array(v['conf']);cnt=np.array(v['counts']);Lam=np.array([[0,1,w],[1,0,w],[1,1,0]])
 check(f'Q2 {key}: columns normalized and 10000 counts',np.allclose(M.sum(0),1) and cnt.sum()==10000)
 check(f'Q2 {key}: risk agrees with confusion counts',np.isclose((cnt*Lam).sum()/10000,v['risk']))
r3=json.loads((root/'out/q3_results.json').read_text())
for key,n in [('wine',4898),('har',10299)]:
 v=r3[key];cnt=np.array(v['counts']);check(f'Q3 {key}: error and normalized confusion agree',cnt.sum()==n and np.isclose(1-np.trace(cnt)/n,v['perr']) and np.allclose(np.array(v['conf']).sum(0),1))
 check(f'Q3 {key}: positive regularization per class',all(c['lam']>0 for c in v['per_class'].values()))
# Inspect executable preprocessing, not just saved values.
s=(root/'q3.py').read_text()
check('Q3 CV scaling is fitted on fold training data','center, scale = train.mean(0), train.std(0)' in s)
check('Q3 official-split scaling is fitted on training data','center, scale = Xh[:ntr].mean(0), Xh[:ntr].std(0)' in s)
(root/'out/verification.json').write_text(json.dumps({'passed':len(checks),'checks':checks},indent=2))
print(f'{len(checks)} checks passed')
