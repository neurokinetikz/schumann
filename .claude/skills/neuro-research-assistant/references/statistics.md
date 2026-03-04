# Statistical Methods Reference

## Test Selection Decision Tree

### Step 1: Identify the Research Question
- **Comparison**: Are groups/conditions different?
- **Association**: Are variables related?
- **Prediction**: Can we predict outcomes?

### Step 2: Check Data Structure
- **Number of groups**: 2 vs >2
- **Design**: Independent vs paired/repeated
- **Variables**: Continuous vs categorical vs ordinal

### Step 3: Check Assumptions
```python
from scipy import stats
import numpy as np

# Normality (for n < 50)
stat, p = stats.shapiro(data)
normal = p > 0.05

# Homogeneity of variance (for 2+ groups)
stat, p = stats.levene(group1, group2)
homogeneous = p > 0.05

# Sample size adequacy
n_adequate = len(data) >= 20  # rule of thumb
```

---

## Parametric Tests

### Two Groups

**Independent Samples t-test**
```python
from scipy import stats

t, p = stats.ttest_ind(group1, group2)

# With unequal variances (Welch's)
t, p = stats.ttest_ind(group1, group2, equal_var=False)

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) +
                      (len(group2)-1)*np.var(group2, ddof=1)) /
                     (len(group1) + len(group2) - 2))
d = (np.mean(group1) - np.mean(group2)) / pooled_std
```

**Paired Samples t-test**
```python
t, p = stats.ttest_rel(condition1, condition2)

# Effect size (Cohen's d for paired)
diff = condition1 - condition2
d = np.mean(diff) / np.std(diff, ddof=1)
```

### Multiple Groups

**One-way ANOVA**
```python
F, p = stats.f_oneway(group1, group2, group3)

# Effect size (eta-squared)
ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
ss_total = sum((x - grand_mean)**2 for g in groups for x in g)
eta_sq = ss_between / ss_total
```

**Repeated Measures ANOVA**
```python
import pingouin as pg

# Data in long format
aov = pg.rm_anova(data=df, dv='value', within='condition', subject='subject')
```

### Correlations

**Pearson Correlation**
```python
r, p = stats.pearsonr(x, y)

# 95% CI via Fisher z-transform
z = np.arctanh(r)
se = 1 / np.sqrt(len(x) - 3)
ci_z = (z - 1.96*se, z + 1.96*se)
ci_r = np.tanh(ci_z)
```

---

## Non-Parametric Tests

### Two Groups

**Mann-Whitney U (independent)**
```python
U, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')

# Effect size (rank-biserial correlation)
n1, n2 = len(group1), len(group2)
r = 1 - (2*U) / (n1 * n2)
```

**Wilcoxon Signed-Rank (paired)**
```python
stat, p = stats.wilcoxon(condition1, condition2)

# Effect size (matched-pairs rank-biserial)
# r = Z / sqrt(N)
```

### Multiple Groups

**Kruskal-Wallis (independent)**
```python
H, p = stats.kruskal(group1, group2, group3)

# Post-hoc: Dunn's test with correction
import scikit_posthocs as sp
p_matrix = sp.posthoc_dunn([group1, group2, group3], p_adjust='fdr_bh')
```

**Friedman Test (repeated)**
```python
stat, p = stats.friedmanchisquare(cond1, cond2, cond3)

# Post-hoc: Nemenyi or Wilcoxon with correction
```

### Correlations

**Spearman Rank Correlation**
```python
rho, p = stats.spearmanr(x, y)
```

**Kendall's Tau**
```python
tau, p = stats.kendalltau(x, y)
```

---

## Permutation and Bootstrap Methods

### When to Use
- Small sample sizes (N < 20)
- Non-normal distributions
- Complex test statistics without known distributions
- Spatiotemporal data with dependencies

### Basic Permutation Test
```python
def permutation_test(group1, group2, n_perm=10000, stat_func=np.mean):
    observed = stat_func(group1) - stat_func(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    null_dist = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = stat_func(combined[:n1]) - stat_func(combined[n1:])
        null_dist.append(perm_diff)

    p_value = np.mean(np.abs(null_dist) >= np.abs(observed))
    return observed, p_value, null_dist
```

### Bootstrap Confidence Intervals
```python
def bootstrap_ci(data, stat_func=np.mean, n_boot=10000, ci=0.95):
    boot_stats = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(sample))

    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [100*alpha, 100*(1-alpha)])
```

### Cluster-Based Permutation (EEG)
```python
import mne
from mne.stats import permutation_cluster_test

# For time-frequency or spatiotemporal data
T_obs, clusters, cluster_p, H0 = permutation_cluster_test(
    [condition1_data, condition2_data],
    n_permutations=1000,
    threshold=dict(start=0, step=0.2),  # or fixed t-value
    tail=0,  # two-tailed
    out_type='mask',
    n_jobs=-1
)
```

---

## Multiple Comparisons Correction

### FWER Control (Family-Wise Error Rate)

**Bonferroni** - Most conservative
```python
alpha_corrected = 0.05 / n_tests
# or
p_corrected = min(p * n_tests, 1.0)
```

**Holm-Sidak** - Less conservative, still controls FWER
```python
from statsmodels.stats.multitest import multipletests
reject, p_corrected, _, _ = multipletests(p_values, method='holm-sidak')
```

### FDR Control (False Discovery Rate)

**Benjamini-Hochberg** - Standard FDR
```python
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

**Benjamini-Yekutieli** - For dependent tests (connectivity)
```python
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_by')
```

### Cluster-Based Permutation

Best for spatiotemporal EEG data:
- Controls FWER at cluster level
- Exploits spatial/temporal autocorrelation
- MNE implementation handles electrode adjacency

```python
# Define adjacency (for EEG sensors)
adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')

# Run cluster test
F_obs, clusters, cluster_p, H0 = mne.stats.spatio_temporal_cluster_test(
    X,  # shape: (n_subjects, n_times, n_channels)
    adjacency=adjacency,
    n_permutations=1000
)
```

### When to Use What

| Scenario | Method |
|----------|--------|
| Few planned comparisons (<5) | Bonferroni |
| Many exploratory comparisons | FDR (BH) |
| Dependent tests (connectivity) | FDR (BY) |
| EEG time/frequency/space | Cluster permutation |
| Very conservative needed | Bonferroni or Holm |

---

## Effect Size Calculation

### Standardized Mean Differences

**Cohen's d (independent groups)**
```python
def cohens_d_independent(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

**Cohen's d (paired/repeated)**
```python
def cohens_d_paired(condition1, condition2):
    diff = condition1 - condition2
    return np.mean(diff) / np.std(diff, ddof=1)
```

**Hedges' g (small sample correction)**
```python
def hedges_g(d, n1, n2):
    df = n1 + n2 - 2
    correction = 1 - (3 / (4*df - 1))
    return d * correction
```

### Variance Explained

**Eta-squared (ANOVA)**
```python
eta_sq = ss_effect / ss_total
```

**Partial eta-squared**
```python
partial_eta_sq = ss_effect / (ss_effect + ss_error)
```

**Omega-squared (less biased)**
```python
omega_sq = (ss_effect - df_effect * ms_error) / (ss_total + ms_error)
```

### Correlation Effect Sizes

| r | Interpretation |
|---|----------------|
| 0.10 | Small |
| 0.30 | Medium |
| 0.50 | Large |

### Confidence Intervals for Effect Sizes
```python
# For Cohen's d
from scipy.stats import nct

def d_ci(d, n1, n2, ci=0.95):
    df = n1 + n2 - 2
    ncp = d * np.sqrt(n1 * n2 / (n1 + n2))  # non-centrality parameter
    t_crit_lo = nct.ppf((1-ci)/2, df, ncp)
    t_crit_hi = nct.ppf(1-(1-ci)/2, df, ncp)
    se = np.sqrt(n1 + n2) / np.sqrt(n1 * n2)
    return (t_crit_lo * se, t_crit_hi * se)
```

---

## Power Analysis

### A Priori Power Analysis
```python
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()

# Find required N for desired power
n = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)

# Find achieved power with given N
power = analysis.power(effect_size=0.5, nobs1=30, alpha=0.05)
```

### Rules of Thumb for EEG

| Analysis | Minimum N | Notes |
|----------|-----------|-------|
| Group comparison | 20-30/group | For medium effects |
| Within-subject | 15-20 | Paired designs more powerful |
| Correlation | 30+ | For r=0.3 detection |
| Cluster permutation | 12+ | Depends on effect size |

### Post-Hoc Power (Caveats)
- Post-hoc power from observed effect is circular
- Better: report effect size + CI
- Or: sensitivity analysis (what effect could we detect?)

---

## Bayesian Methods

### When to Use Bayesian
- Want to quantify evidence FOR null hypothesis
- Prior knowledge to incorporate
- Sequential testing / optional stopping
- Small samples where priors matter

### Bayes Factors Interpretation

| BF₁₀ | Evidence for H1 |
|------|-----------------|
| 1-3 | Anecdotal |
| 3-10 | Moderate |
| 10-30 | Strong |
| 30-100 | Very strong |
| >100 | Extreme |

| BF₀₁ | Evidence for H0 |
|------|-----------------|
| 1-3 | Anecdotal |
| 3-10 | Moderate |
| >10 | Strong |

### Bayesian t-test
```python
import pingouin as pg

# Returns BF10 (evidence for alternative)
result = pg.ttest(group1, group2, paired=False)
bf = result['BF10'].values[0]
```

### Bayesian Correlation
```python
result = pg.corr(x, y, method='pearson')
bf = result['BF10'].values[0]
```

### Credible Intervals vs Confidence Intervals
- **Confidence Interval**: 95% of such intervals contain true value (frequentist)
- **Credible Interval**: 95% probability true value is in interval (Bayesian)
- Credible intervals often more intuitive to interpret

---

## Reporting Guidelines

### Minimum Reporting
1. Test statistic and degrees of freedom
2. Exact p-value (not just p < 0.05)
3. Effect size with CI
4. Sample sizes per group
5. Correction method if multiple tests

### Example Report
```
Ignition windows showed significantly higher theta power than baseline
(t(24) = 3.45, p = .002, d = 0.71, 95% CI [0.28, 1.14]).
Multiple comparisons across 14 electrodes were corrected using FDR
(q = 0.05), with 8 electrodes showing significant effects.
```

### Common Mistakes
- Reporting p < 0.05 without effect size
- Not correcting for multiple comparisons
- Claiming "no difference" from non-significant result
- Using post-hoc power to interpret null results
- Misinterpreting confidence intervals
