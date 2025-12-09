# shape_vs_resonance

**Total functions:** 21 (18 public, 3 private)

## Public Functions

### `ensure_dir(d)`

### `ensure_timestamp_column(df, time_col, default_fs)`

### `infer_fs(df, time_col)`

### `get_series(df, name)`

### `cycles_morphology(x, fs, f0, half, sharp_win)`
> Compute cycle features from bandpassed x around f0±half.

### `next_idx(arr, i)`

### `irasa_psd(x, fs, hset, nperseg, fmax)`
> Approximate IRASA: resample by h and 1/h, compute PSDs, map freqs back by /h and *h,

### `res_psd(y, up, down)`

### `bicoherence_discrete_auto(x, fs, f_list, nperseg, step)`
> Auto‑bicoherence on a discrete frequency list f_list (Hz). Returns matrix B[i,j] for f1=f_list[i], f2=f_list[j] with f1+f2 in grid.

### `bin_idx(f)`

### `bicoherence_discrete_cross(sr, eeg, fs, f_list, nperseg, ...)`
> Cross‑bicoherence variant: B_sse(f1,f2) = <S(f1) S(f2) E*(f1+f2)> / sqrt(<|S(f1)S(f2)|^2><|E(f1+f2)|^2>).

### `bin_idx(f)`

### `analyze_shape_vs_resonance(RECORDS, eeg_channel: str, sr_channel: str, time_col, fundamental, ...)`
> Run morphology, IRASA, and (cross‑)bicoherence tests; save figures and a summary CSV.

### `at(freq, bw)`

### `sur_cross(nr)`

### `sur_auto(nr)`

### `nearest_idx(arr, val)`

### `heat(M, thr, title, fname)`

## Private/Helper Functions

- `_bandpass(x, fs, lo, ...)`
- `_lowpass(x, fs, hi, ...)`
- `_fft_segments(x, fs, nperseg, ...)`
