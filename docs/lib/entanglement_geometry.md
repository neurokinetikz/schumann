# entanglement_geometry

**Total functions:** 13 (4 public, 9 private)

## Public Functions

### `plot_entanglement_geometry_deltas(df: pd.DataFrame, by_session: bool, session_col: str) -> None`
> Bar plot of deltas per band. If by_session=True, aggregate by band and use mean±sem.

### `plot_entanglement_geometry_levels(df: pd.DataFrame) -> None`
> Side-by-side bars for ignition vs baseline values (min-cut, entropy, PLV) per band.

### `plot_entanglement_geometry_scatter(df: pd.DataFrame) -> None`
> Scatter plots to visualize relationships across bands: Δmincut vs ΔPLV, Δentropy vs ΔPLV, etc.

### `run_entanglement_geometry_minCut_PLV(x_base: Optional[ArrayLike], x_ign: Optional[ArrayLike], fs: Optional[float], bands: BandDict) -> Dict[str, Union[pd.DataFrame, Dict[str, ArrayLike]]]`
> Compute per-band network metrics linking *entanglement-geometry* and synchrony:

## Private/Helper Functions

- `_err_sem(x: np.ndarray)`
- `_butter_bandpass(sig: ArrayLike, fs: float, f_lo: float, ...)`
- `_analytic_hilbert(sig: ArrayLike)`
- `_compute_plv_matrix(analytic: ArrayLike)`
- `_ensure_symmetric(W: ArrayLike)`
- `_laplacian_entropy(W: ArrayLike, eps: float)`
- `_global_min_cut_weight(W: ArrayLike, edge_min: float)`
- `_mean_upper_triangle(W: ArrayLike)`
- `_plv_from_timeseries(x: ArrayLike, fs: float, band: Tuple[float, float], ...)`
