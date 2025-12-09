# directed_connectivity

**Total functions:** 8 (5 public, 3 private)

## Public Functions

### `df_to_raw(records: pd.DataFrame, ch_names: List[str], sfreq: float, montage: str) -> mne.io.Raw`
> Convert a (Timestamp + EEG.<ch>.FILTERED) DataFrame into MNE RawArray.

### `phase_transfer_entropy_stub() -> float`
> Stub for PTE (wire IDTxl/JIDT here if desired).

### `sensor_directed_connectivity(raw: mne.io.BaseRaw, windows: List[Tuple[float, float]], bands: Dict[str, Tuple[float, float]], source_ch: str, posterior_chs: Tuple[str, ...], ...) -> pd.DataFrame`
> Compute dPLI + (pairwise) Granger from F4→posterior sensors per event window & band.

### `source_directed_connectivity(raw: mne.io.BaseRaw, windows: List[Tuple[float, float]], subjects_dir: str, subject: str, trans: str, ...) -> pd.DataFrame`
> Compute dPLI + conditional Granger: DLPFC_R→(occipital/temporal/parietal) per window & band.

### `run_topdown_ignition_pipeline(records: Optional[pd.DataFrame], electrodes: Optional[List[str]], fs: Optional[float], raw: Optional[mne.io.BaseRaw], windows: List[Tuple[float, float]], ...) -> pd.DataFrame`
> Run SENSOR or SOURCE pipeline.

## Private/Helper Functions

- `_compute_dpli(data2xT: np.ndarray, sfreq: float, fmin: float, ...)`
- `_conditional_granger(source: np.ndarray, target: np.ndarray, conditioners: Optional[np.ndarray], ...)`
- `_pick(labels_list: List[str])`
