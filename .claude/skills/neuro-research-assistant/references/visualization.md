# Visualization Patterns Reference

## Plot Type Selection

| Data Type | Primary Plot | Alternatives |
|-----------|-------------|--------------|
| Single time series | Line plot | Area plot |
| Multi-channel EEG | Stacked traces | Butterfly plot |
| Power spectrum | Log-log PSD | Linear PSD |
| Time-frequency | Heatmap | Contour |
| Connectivity matrix | Heatmap | Circular graph |
| Scalp topography | Topomap | 3D head |
| Distributions | Violin + points | Box, histogram |
| Group comparisons | Bar + error + points | Paired lines |

---

## Time Series Plots

### Multi-Channel Stacked Plot
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_stacked_eeg(data, times, channels, spacing=None, ax=None):
    """Plot EEG channels stacked vertically."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    n_channels = len(channels)
    if spacing is None:
        spacing = np.max(np.abs(data)) * 2

    for i, (ch, trace) in enumerate(zip(channels, data)):
        offset = i * spacing
        ax.plot(times, trace + offset, 'k', lw=0.5)
        ax.text(times[0] - 0.02*(times[-1]-times[0]), offset,
                ch, ha='right', va='center', fontsize=8)

    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    ax.set_xlim(times[0], times[-1])
    return ax
```

### Event Shading
```python
def shade_events(ax, events, color='yellow', alpha=0.3, label=None):
    """Add vertical shading for event windows."""
    for i, (start, end) in enumerate(events):
        ax.axvspan(start, end, color=color, alpha=alpha,
                   label=label if i == 0 else None)
```

### Vertical Event Markers
```python
def mark_events(ax, times, color='red', linestyle='--', alpha=0.7):
    """Add vertical lines at event times."""
    for t in times:
        ax.axvline(t, color=color, linestyle=linestyle, alpha=alpha)
```

---

## Spectral Plots

### PSD with Confidence Band
```python
def plot_psd_with_ci(freqs, psd_mean, psd_low, psd_high, ax=None,
                     log_scale=True, harmonic_freqs=None):
    """PSD plot with confidence band and harmonic annotations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(freqs, psd_low, psd_high, alpha=0.3, color='blue')
    ax.plot(freqs, psd_mean, 'b-', lw=1.5)

    if log_scale:
        ax.set_yscale('log')
        ax.set_xscale('log')

    if harmonic_freqs:
        for f in harmonic_freqs:
            ax.axvline(f, color='red', linestyle=':', alpha=0.5)
            ax.text(f, ax.get_ylim()[1], f'{f:.1f}', ha='center',
                    va='bottom', fontsize=8, color='red')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (μV²/Hz)')
    return ax
```

### 1/f Slope Visualization
```python
def plot_1f_slope(freqs, psd, fit_range=(1, 30), ax=None):
    """Plot PSD with 1/f fit line."""
    if ax is None:
        fig, ax = plt.subplots()

    # Log-log plot
    ax.loglog(freqs, psd, 'b-', lw=1)

    # Fit 1/f in range
    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    slope, intercept = np.polyfit(log_f, log_p, 1)

    # Plot fit
    fit_line = 10**(intercept + slope * np.log10(freqs[mask]))
    ax.loglog(freqs[mask], fit_line, 'r--', lw=2,
              label=f'1/f slope: {-slope:.2f}')

    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    return ax
```

---

## Time-Frequency Plots

### TF Heatmap
```python
def plot_tf_heatmap(power, times, freqs, ax=None, cmap='viridis',
                    vmin=None, vmax=None, cbar_label='Power'):
    """Time-frequency heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.pcolormesh(times, freqs, power, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    return ax, im
```

### Diverging Colormap for Contrasts
```python
def plot_tf_contrast(power_diff, times, freqs, ax=None):
    """TF plot with diverging colormap centered at zero."""
    if ax is None:
        fig, ax = plt.subplots()

    vmax = np.max(np.abs(power_diff))
    im = ax.pcolormesh(times, freqs, power_diff,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='auto')
    plt.colorbar(im, ax=ax, label='Difference')
    return ax
```

### Cone of Influence
```python
def add_coi(ax, times, freqs, fs, n_cycles):
    """Add cone of influence shading for wavelet analysis."""
    # COI: frequencies where edge effects are significant
    coi_freqs = n_cycles / (2 * np.pi * np.minimum(times, times[-1] - times))
    ax.fill_between(times, 0, coi_freqs, alpha=0.2, color='gray')
```

---

## Topomaps

### Basic Topomap (MNE)
```python
import mne

def plot_topomap(values, info, ax=None, cmap='RdBu_r', vmin=None, vmax=None):
    """Plot scalp topography."""
    if ax is None:
        fig, ax = plt.subplots()

    mne.viz.plot_topomap(values, info, axes=ax, cmap=cmap,
                         vlim=(vmin, vmax), show=False)
    return ax
```

### Multi-Band Topomap Grid
```python
def plot_topomap_grid(band_data, info, bands, figsize=(12, 3)):
    """Grid of topomaps for multiple frequency bands."""
    n_bands = len(bands)
    fig, axes = plt.subplots(1, n_bands, figsize=figsize)

    for ax, (band_name, data) in zip(axes, band_data.items()):
        mne.viz.plot_topomap(data, info, axes=ax, show=False)
        ax.set_title(band_name)

    plt.tight_layout()
    return fig
```

### Symmetric Colorbar for Differences
```python
def symmetric_clim(data):
    """Get symmetric color limits centered at zero."""
    vmax = np.max(np.abs(data))
    return -vmax, vmax
```

---

## Connectivity Matrices

### Heatmap with Labels
```python
def plot_connectivity_matrix(matrix, labels, ax=None, cmap='viridis',
                             mask_diagonal=True):
    """Connectivity matrix heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if mask_diagonal:
        matrix = matrix.copy()
        np.fill_diagonal(matrix, np.nan)

    im = ax.imshow(matrix, cmap=cmap, aspect='equal')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    plt.colorbar(im, ax=ax)
    return ax
```

### Circular Graph Layout
```python
import networkx as nx

def plot_connectivity_circular(matrix, labels, threshold=None, ax=None):
    """Circular graph layout for connectivity."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    G = nx.Graph()
    G.add_nodes_from(labels)

    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i < j:
                weight = matrix[i, j]
                if threshold is None or weight > threshold:
                    G.add_edge(l1, l2, weight=weight)

    pos = nx.circular_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    nx.draw(G, pos, ax=ax, with_labels=True, node_size=500,
            edge_color=weights, edge_cmap=plt.cm.viridis,
            width=2, font_size=8)
    return ax
```

---

## Distribution Plots

### Violin + Individual Points
```python
import seaborn as sns

def plot_violin_points(data, x, y, ax=None, palette='Set2'):
    """Violin plot with individual data points."""
    if ax is None:
        fig, ax = plt.subplots()

    sns.violinplot(data=data, x=x, y=y, ax=ax, palette=palette, alpha=0.7)
    sns.stripplot(data=data, x=x, y=y, ax=ax, color='black',
                  alpha=0.5, size=4, jitter=True)
    return ax
```

### Paired Lines Plot
```python
def plot_paired_lines(pre, post, labels=('Pre', 'Post'), ax=None):
    """Paired comparison with connected lines."""
    if ax is None:
        fig, ax = plt.subplots()

    for p, q in zip(pre, post):
        ax.plot([0, 1], [p, q], 'o-', color='gray', alpha=0.5)

    ax.errorbar([0, 1], [np.mean(pre), np.mean(post)],
                yerr=[np.std(pre)/np.sqrt(len(pre)),
                      np.std(post)/np.sqrt(len(post))],
                fmt='s-', color='red', markersize=10, capsize=5, lw=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.3, 1.3)
    return ax
```

---

## Publication Standards

### Figure Sizing

| Format | Width (inches) | Common Heights |
|--------|----------------|----------------|
| Single column | 3.5 | 2.5-4 |
| 1.5 column | 5.5 | 3-5 |
| Double column | 7.0 | 4-8 |
| Full page | 7.0 | 9 |

### Font Sizes

| Element | Minimum Size |
|---------|--------------|
| Axis labels | 10 pt |
| Tick labels | 8 pt |
| Legend | 8 pt |
| Panel labels (A, B, C) | 12 pt, bold |
| Title | 11 pt |

### DPI Requirements

| Use | DPI |
|-----|-----|
| Screen | 72-150 |
| Print draft | 150 |
| Publication | 300+ |
| Line art | 600+ |

### Standard RC Params
```python
publication_params = {
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

plt.rcParams.update(publication_params)
```

### Colormap Selection

| Data Type | Recommended Colormaps |
|-----------|-----------------------|
| Sequential (low→high) | viridis, plasma, cividis |
| Diverging (neg↔pos) | RdBu_r, coolwarm, seismic |
| Categorical | tab10, Set2, Dark2 |
| Perceptually uniform | viridis, cividis, inferno |

**Avoid**: jet, rainbow (not perceptually uniform)

### Accessibility

- Use colorblind-friendly palettes (viridis, cividis)
- Add texture/patterns in addition to color
- Ensure sufficient contrast
- Label directly when possible (not just legend)

---

## Multi-Panel Figures

### Subplot Grid
```python
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()  # Easy iteration

for i, ax in enumerate(axes):
    # Plot on each panel
    ax.text(0.05, 0.95, f'({chr(65+i)})', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
```

### GridSpec for Irregular Layouts
```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 3, figure=fig)

ax_large = fig.add_subplot(gs[0, :2])  # Top-left, 2 columns
ax_small1 = fig.add_subplot(gs[0, 2])  # Top-right
ax_small2 = fig.add_subplot(gs[1, 0])  # Bottom-left
ax_small3 = fig.add_subplot(gs[1, 1])  # Bottom-middle
ax_small4 = fig.add_subplot(gs[1, 2])  # Bottom-right
```

### Shared Axes
```python
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)

axes[-1].set_xlabel('Time (s)')
```
