# Assignment 1: learn the ideas through your experiments

Use this alongside `EECE5644_Assignment1_Reviewed.pdf`. The PDF contains the complete mathematical results and code; this guide explains how to reason about them. You do not need EM to finish this assignment: Q2 supplies the mixture parameters, and Q3 fits single Gaussians by sample averages. MAP class decisions here are also different from MAP parameter estimation in the October 13 lecture.

## 1. The recurring problem

You observe a feature vector x and must predict a hidden class label L. Your output is D. In Q1, x has four numbers and L is 0 or 1. In Q2, x has three numbers and L is 1, 2, or 3. In Q3, the numbers are real measurements of wine or human activity.

Three quantities answer different questions:

- Prior P(L=j): how common is class j before you examine x?
- Likelihood density p(x|L=j): how compatible is this observation with the class model?
- Posterior P(L=j|x): after seeing x, how probable is class j?

Bayes' rule connects them:

P(L=j|x) = p(x|L=j) P(L=j) / sum_k[p(x|L=k) P(L=k)].

A continuous density is not a point probability. It can exceed one; posterior probabilities remain between zero and one and sum to one across classes.

### Worked example from our earlier lesson

Suppose p(x|0)=0.10 and p(x|1)=0.08. Multiply by the priors:

- Class 0 joint score: 0.10*0.35 = 0.035.
- Class 1 joint score: 0.08*0.65 = 0.052.
- Posterior class 1: 0.052/(0.035+0.052) = 0.5977.

Predict class 1 under equal error costs. If the likelihoods were exactly equal, the posteriors would equal the priors: class 1 would have posterior 0.65.

## 2. Why expected risk is more general than accuracy

Let lambda(i,j) be the cost of predicting i when j is true. Its first index is the decision, its second the true label. For a fixed observation, the expected cost of deciding i is

R(i|x) = sum_j lambda(i,j) P(L=j|x).

Choose the decision with the smallest conditional risk. This minimizes population expected risk because each observation's contribution is minimized separately.

Under 0-1 loss, correct decisions cost zero and incorrect ones cost one. Then

R(i|x) = 1 - P(L=i|x).

Minimizing risk becomes maximizing the posterior: the MAP classifier. MAP here estimates a **class label**, not a distribution parameter. MLE and MAP parameter estimation are separate uses of likelihood and Bayes' rule.

### Why unequal costs matter

Suppose posteriors for classes 1, 2, and 3 are (0.80, 0.15, 0.05).

Under 0-1 loss, choose class 1.

With class-3 missed-detection penalty w=10:

- R(1|x)=0.15+10*0.05=0.65.
- R(2|x)=0.80+10*0.05=1.30.
- R(3|x)=0.80+0.15=0.95.

Still choose class 1.

With w=100, the risks are 5.15, 5.80, and 0.95. Choose class 3 even though it has the smallest posterior. That decision protects against a rare but very expensive mistake. This explains the behavior in Q2.

## 3. Q1: generate one fair comparison dataset

First randomly choose 10,000 labels with probabilities (0.35,0.65). For each label draw a Gaussian observation using the corresponding mean and covariance. The mean is the center of the point cloud; the covariance describes spread and linear association between coordinates.

The fixed seed gives 3460 class-0 observations and 6540 class-1 observations. The counts are random; priors do not require exactly 3500 and 6500. All three classifiers must use the same generated observations.

Code entry point: `sample()` in q1.py. `out/q1_data.npz` contains X and L. X has shape (10000,4); L has shape (10000,).

## 4. Q1A: derive the likelihood-ratio threshold

Write the two risks:

R(0|x)=lambda00*P(0|x)+lambda01*P(1|x).
R(1|x)=lambda10*P(0|x)+lambda11*P(1|x).

Choose 1 if R(1|x)<R(0|x). Move terms:

(lambda01-lambda11) P(1|x) > (lambda10-lambda00) P(0|x).

Substitute Bayes' rule, cancel the common marginal density, and divide when lambda01>lambda11:

p(x|1)/p(x|0) > [(lambda10-lambda00)*P(0)] / [(lambda01-lambda11)*P(1)].

For 0-1 loss, gamma*=0.35/0.65=0.53846.

A smaller threshold makes it easier to predict 1. A larger threshold makes it harder. Increasing the penalty for missing class 1 increases the denominator and lowers the threshold.

If arbitrary loss differences are zero or negative, use the undivided risk inequality. Dividing by a negative number reverses the inequality.

## 5. What the Gaussian formula is doing

A d-dimensional Gaussian log-density is

log g(x|m,C) = -0.5*[d*log(2*pi) + log(det(C)) + (x-m)^T C^(-1) (x-m)].

The final term is squared Mahalanobis distance: distance measured relative to the cloud's spread and orientation. Moving along a naturally wide direction is less surprising than moving the same amount along a narrow direction. The determinant term accounts for the cloud's volume: broader distributions spread their density over more space.

Subtract the two log-densities to form the log-likelihood ratio. Compare it with log(gamma). Logs preserve ordering, avoid tiny-density underflow, and turn products into sums.

Code entry point: `gauss_logpdf()` in common.py. It uses a Cholesky factor, not an explicit matrix inverse.

## 6. ROC curves: see every threshold at once

Define:

- True positive: L=1, D=1.
- False positive: L=0, D=1.
- False negative: L=1, D=0.
- True negative: L=0, D=0.

TPR = TP/(TP+FN). FPR = FP/(FP+TN). The denominators differ because each rate is conditional on a different true class.

At threshold -infinity on the log score, everything is predicted 1, so (FPR,TPR)=(1,1). At the maximum observed score, the strict rule score>threshold predicts no positives, so the point is (0,0).

Increasing the threshold takes the curve from (1,1) toward (0,0). A good curve stays near the upper-left corner: high detection, few false alarms. AUC is an overall ranking measure; it does not itself determine the minimum-cost threshold.

For each threshold in Q1A/B:

Estimated error = 0.35*FPR + 0.65*(1-TPR).

Example: TPR=0.97 and FPR=0.02 give error 0.35*0.02+0.65*0.03=0.0265.

The empirical minimum threshold adapts to the particular dataset. It need not equal the population-optimal threshold on 10K observations. Its minimum measured error is selected on the same data, so it is optimistically selected.

Code entry points: `roc_sweep()` and `auc()` in common.py; `analyse()` in q1.py. Tied scores must be processed together. Preserving vertical-segment order matters when integrating the ROC.

## 7. Q1B: what Naive Bayes loses

For a multivariate Gaussian, diagonal covariance implies conditional independence of the features. Replace the true covariance matrices by diagonal matrices with the same diagonal entries. Do not regenerate data from these diagonal models: that would change the problem.

We deliberately evaluate an incorrect model on the original correlated data. Its best empirical error rises from about 2.59% to 3.25%. It still uses distinct class variances, so its decision boundary can remain quadratic.

The prior-derived threshold is optimal for the assumed diagonal model. It is not guaranteed to be optimal when the data actually follow the full-covariance model.

## 8. Q1C: Fisher LDA compresses four numbers into one

Estimate each class mean and covariance from the generated data. Set

SW=C0_hat+C1_hat.
SB=(m1_hat-m0_hat)(m1_hat-m0_hat)^T.

Fisher's criterion is J(w)=(w^T SB w)/(w^T SW w).

The numerator rewards separation of projected class means; the denominator penalizes within-class spread. For example, a direction that separates the centers by 10 units is not necessarily useful if each class has a projected standard deviation of 100 units.

The maximizing direction solves SB*w=lambda*SW*w. Select the largest eigenvalue. For two classes, the direction is also proportional to inverse(SW)*(m1_hat-m0_hat). Normalize and choose its sign so class 1 projects higher.

Here w is approximately (0.6028,0.2747,0.7086,0.2432). Each point becomes z=w^T x. Sweep the scalar threshold tau and classify 1 when z>tau.

LDA's error is about 5.21%. It uses one straight separating hyperplane, while the Bayes boundary is curved because the class covariances differ. Fisher LDA does not generally minimize classification error for arbitrary class distributions.

Important distinction: PCA maximizes variance without using labels. Fisher LDA uses labels to favor separation. The assignment asks for equal class weights in the scatter matrices, even though the priors are unequal.

## 9. Q2: a mixture component is not always a class

There are four Gaussian components but only three classes. Components 1 and 2 are classes 1 and 2. Components 3 and 4 both belong to class 3, with equal mixing weights.

Therefore p(x|3)=0.5*g3(x)+0.5*g4(x). Compute the sum of densities before taking its logarithm. Averaging the two log-densities is a different quantity and would be incorrect.

Generate one 10K dataset and reuse it for all loss matrices. The tetrahedral means are pairwise 2.5 units apart, approximately 2.52 times the chosen average component standard deviation.

For MAP: maximize log likelihood plus log prior.
For general ERM: compute posteriors, calculate all conditional risks, and choose the lowest.

Code entry point: `class_loglik()` in q2.py. `post @ Lam.T` computes each decision's conditional risk. The transpose is necessary because the loss matrix has decision rows and truth columns.

## 10. Read the confusion matrix correctly

Entry (i,j) estimates P(D=i|L=j). Fix a true-label column j; divide all counts in that column by the number of true class-j observations. Each column sums to one. The diagonal gives recall for each class.

Overall error cannot generally be obtained by an unweighted average of off-diagonal probabilities. Weight class-conditional error rates by class priors or use the total number of wrong observations divided by N.

For a count matrix C with decision rows and truth columns:

Observed error = 1 - trace(C)/N.
Observed risk = sum_ij lambda(i,j)*C(i,j)/N.

For 0-1 loss these agree. For Q2's unequal losses they differ substantially.

## 11. Q3: learn distributions from observations

Unlike Q1 and Q2, the true probability distributions are unknown. For each class j, estimate:

- Prior: Nj/N.
- Mean: average of class-j feature vectors.
- Covariance: average outer product of deviations, with denominator Nj.

The denominator Nj is the Gaussian maximum-likelihood/sample-average estimator used here; Nj-1 is the commonly used unbiased covariance estimator.

Assume a single Gaussian per class and predict the largest log-density plus log prior. Different class covariances produce quadratic discriminant analysis, often called QDA. This is not the Fisher projection classifier from Q1C.

## 12. Why covariance regularization is necessary

A covariance matrix needs an inverse and positive determinant for a full-dimensional Gaussian density. It becomes singular when features are linearly dependent or there are too few class observations relative to dimension.

Wine quality 9 has 5 observations in 11 dimensions. After centering, covariance rank is at most 5-1=4. A full 11D inverse is impossible without modification.

Add lambda*I. This adds lambda to every eigenvalue and gives a positive variance floor. The assignment suggests lambda=alpha*trace(C)/rank(C), a small fraction of the mean nonzero eigenvalue. In floating-point calculations use numerical rank with a tolerance.

Standardize features before adding isotropic regularization. Otherwise a single lambda can overwhelm low-variance features measured in small units. Fit standardization only on training observations whenever evaluating on held-out data.

## 13. Training error versus validation error

The assignment explicitly requests training on all samples and then scoring those same samples. Follow that instruction and clearly call the results training errors:

- Wine: about 50.33% error.
- HAR: about 0.301% error.

Those are not estimates of independent future-sample performance. In this rerun, wine CV error is about 51.49% and HAR validation error about 8.18%.

The HAR official split was used to select alpha. It therefore serves as validation, even though the dataset names its files train and test. Repeatedly choosing the best alpha on it means its final score is not an untouched test score.

Why can HAR training error be so low? Each class fits a very flexible model in 561 dimensions, and evaluation reuses the fitting observations. Why is wine harder? Neighboring quality classes overlap, labels are imbalanced, and a single Gaussian does not capture all within-class shapes.

## 14. PCA in Question 3

PCA is used for plots, not to reduce the classifier input here. Center the standardized data and use singular value decomposition. Project onto the first two or three principal directions. Explained variance ratios tell you how much total standardized variance each component retains.

Wine PCs 1-3 explain about 29.29%, 14.32%, and 11.11%. HAR PC1 explains about 50.74%. Distinct-looking clusters in a projection do not prove high-dimensional separation; overlapping projected clusters do not prove inseparability in the full space.

## 15. How to explain your results aloud

Q1: 'I compared three scoring rules on exactly the same simulated observations. Correct covariance modeling produced the lowest error, ignoring correlations increased it, and a linear Fisher classifier lost more because the Bayes boundary was quadratic.'

Q2: 'I minimized expected cost, not just mistake count. Increasing the cost of missing class 3 expanded its decision region, improved its recall, and sacrificed accuracy on classes 1 and 2.'

Q3: 'I estimated one Gaussian per class using sample averages, regularized ill-conditioned covariances, and reported the all-sample errors requested. I separated those from validation results and examined whether the Gaussian assumptions were plausible.'

## 16. Self-checks with answers

1. If the false-negative penalty for class 1 increases, what happens to the binary threshold? **It decreases**, favoring more class-1 decisions.
2. Can a classifier have higher error but lower risk than another? **Yes**, under unequal costs.
3. Does diagonal covariance mean features are independent for every distribution? **No**; it means uncorrelated. For a joint Gaussian, uncorrelated coordinates are independent.
4. Why is class-3 density a sum in Q2? **Either component can generate an observation from that class**, so marginalize the hidden component identity.
5. Does a perfect rare-class training result establish strong performance? **No**; it may reflect severe overfitting and tiny sample size.
6. Is MAP classification the same as MAP parameter estimation? **No**; the unknown variable is a class in one case and a model parameter in the other.
7. Is a nominal chi-square Q-Q plot conclusive after fitting and regularization? **No**; its reference distribution is altered by fitting, shrinkage, and rank issues.

## Suggested study sequence

Read sections 1-6, then inspect Q1A's ROC and threshold table. Read sections 7-8 and compare all three Q1 curves. Read sections 9-10 and trace the posterior-risk example by hand. Finish with sections 11-14 and the Q3 tables. Use section 16 to check understanding before reviewing the source appendix.
