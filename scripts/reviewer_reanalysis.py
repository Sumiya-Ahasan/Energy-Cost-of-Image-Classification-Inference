"""Re-analysis for the paper revision (session-block ANOVA, Friedman/Wilcoxon with Holm,
session offsets, paired FP16 intervals). Usage: python reviewer_reanalysis.py all_sessions_results.csv"""
import sys, itertools, numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(sys.argv[1]); d = d[d.valid == True].copy()
d['lg'] = np.log(d.energy_per_sample_j)
y = d['lg'].values
dum = lambda c: pd.get_dummies(d[c], drop_first=True).values.astype(float)
inter = lambda A, B: np.hstack([A[:, [i]] * B[:, [j]] for i in range(A.shape[1]) for j in range(B.shape[1])])
S, M, B, P = dum('session_id'), dum('model'), dum('batch_size'), dum('precision')
blocks = {'Session': S, 'Model': M, 'Batch': B, 'Precision': P, 'Model x Batch': inter(M, B),
          'Model x Precision': inter(M, P), 'Batch x Precision': inter(B, P), 'Model x Batch x Precision': inter(inter(M, B), P)}
def rss(X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; return (r**2).sum(), np.linalg.matrix_rank(X)
X = np.ones((len(d), 1)); prev, pr = rss(X); tot = ((y - y.mean())**2).sum(); rows = []
for k, Z in blocks.items():
    X = np.hstack([X, Z]); r, rk = rss(X); rows.append([k, rk - pr, prev - r]); prev, pr = r, rk
dfres = len(d) - pr; ms = prev / dfres
a = pd.DataFrame(rows, columns=['term', 'df', 'ss'])
a['F'] = a.ss / a.df / ms; a['p'] = stats.f.sf(a.F, a.df, dfres); a['pct_total_ss'] = 100 * a.ss / tot
a.loc[len(a)] = ['Residual', dfres, prev, np.nan, np.nan, 100 * prev / tot]
a.to_csv('anova_with_session_block.csv', index=False)
d['cfg'] = d.model + '|' + d.batch_size.astype(str) + '|' + d.precision
w = d.pivot_table(index='cfg', columns='session_id', values='energy_per_sample_j')
fr = stats.friedmanchisquare(*[w[c] for c in w.columns]); n, k = w.shape
pairs = list(itertools.combinations(w.columns, 2)); ps = np.array([stats.wilcoxon(w[a_], w[b_]).pvalue for a_, b_ in pairs])
o = np.argsort(ps); h = np.zeros(len(ps)); run = 0
for i, j in enumerate(o): run = max(run, (len(ps) - i) * ps[j]); h[j] = min(1, run)
pd.DataFrame({'session_a': [p[0] for p in pairs], 'session_b': [p[1] for p in pairs], 'p_raw': ps, 'p_holm': h}).to_csv('wilcoxon_sessions_holm.csv', index=False)
d['dev'] = d['lg'] - d.groupby('cfg')['lg'].transform('mean')
off = (np.exp(d.groupby('session_id').dev.mean()) - 1) * 100
pd.DataFrame({'session_id': off.index, 'offset_pct': off.values}).to_csv('session_offsets.csv', index=False)
open('friedman_kendall.txt', 'w').write(f"chi2={fr.statistic:.2f} p={fr.pvalue:.3g} n={n} k={k} kendall_W={fr.statistic/(n*(k-1)):.3f}\n")
p = d.pivot_table(index=['model', 'batch_size', 'session_id'], columns='precision', values='energy_per_sample_j').dropna()
p['red'] = 100 * (1 - p.fp16 / p.fp32)
g = p.groupby(level=[0, 1]).red.agg(['mean', 'std', 'count']); g['ci95'] = stats.t.ppf(.975, g['count'] - 1) * g['std'] / np.sqrt(g['count'])
g.to_csv('fp16_reduction_by_model_batch_with_ci.csv')
print(open('friedman_kendall.txt').read()); print(a.round(4)); print(off.round(1).to_dict())
