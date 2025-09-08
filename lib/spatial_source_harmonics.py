"""
Spatial & Source-level Harmonics (0.1–60 Hz)
===========================================
Simple tests + graphs to validate:
  • Topographies per order: map H‑PLI_k (EEG↔SR) and HCS across electrodes
    + Global Field Synchronization (GFS) per harmonic.
  • Source localization (LCMV) at each SR line (if MNE forward model provided).
  • Connectome‑harmonic overlap: project source maps on structural eigenmodes.
  • Network graphs per order: PLV networks at 7.83, 14.3, 20.8, … → modularity, min‑cut, path length.

Dependencies: numpy, scipy, matplotlib, pandas, networkx.
Optional: mne (for topomaps + LCMV), structural connectome (SC) for eigenmodes.

Notes
-----
• H‑PLI_k at an electrode e is |<exp(i(φ_e(fk) − φ_SR(fk)))>| over sliding windows (default 8 s, 1 s step);
  per‑electrode p‑values via circular‑shift surrogates of SR phase.
• HCS (sensor‑level) is a weighted sum across orders of H‑PLI_k per electrode.
• GFS(fk): mean resultant length across channels of phases at fk (spatial phase consensus), averaged in windows.
• All analyses clamp to ≤ 60 Hz and are NaN‑safe.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import networkx as nx

# -------------------------- I/O helpers --------------------------

def ensure_dir(d):
    if d: os.makedirs(d, exist_ok=True)
    return d

def ensure_timestamp_column(df, time_col='Timestamp', default_fs=128.0):
    if time_col in df.columns:
        s = df[time_col]
        if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
            tsec=(pd.to_datetime(s)-pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
            df[time_col] = tsec.values; return time_col
        sn = pd.to_numeric(s, errors='coerce').astype(float)
        if sn.notna().sum()>max(50,0.5*len(df)):
            sn = sn - np.nanmin(sn[np.isfinite(sn)])
            df[time_col] = sn.values; return time_col
    df[time_col] = np.arange(len(df), dtype=float)/default_fs
    return time_col

def infer_fs(df, time_col='Timestamp'):
    t = np.asarray(df[time_col].values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError('Cannot infer fs from time column.')
    return float(1.0/np.median(dt))

# -------------------------- Filtering & analytic --------------------------

def _bandpass(x, fs, lo, hi, order=4):
    ny=0.5*fs; lo=max(1e-6,min(lo,0.99*ny)); hi=max(lo+1e-6,min(hi,0.999*ny))
    b,a=signal.butter(order,[lo/ny, hi/ny], btype='band'); return signal.filtfilt(b,a,x)

def phase_series(x, fs, f0, half):
    xb = _bandpass(x, fs, f0-half, f0+half)
    return np.angle(signal.hilbert(xb))

# -------------------------- Electrode utilities --------------------------

def detect_eeg_channels(df, prefix='EEG.'):
    return [c for c in df.columns if c.startswith(prefix)]

# minimal 2D coords for common 10‑20 labels (fallback if mne not installed)
_COORDS_2D = {
    'Fp1':(-0.5, 1.0),'Fp2':(0.5,1.0),'F7':(-0.9,0.6),'F3':(-0.4,0.6),'Fz':(0,0.7),'F4':(0.4,0.6),'F8':(0.9,0.6),
    'FC5':(-0.7,0.4),'FC6':(0.7,0.4),'T7':(-1.0,0.2),'C3':(-0.5,0.2),'Cz':(0,0.2),'C4':(0.5,0.2),'T8':(1.0,0.2),
    'TP9':(-1.1,-0.1),'CP5':(-0.7,0.0),'CP6':(0.7,0.0),'TP10':(1.1,-0.1),'P7':(-0.9,-0.2),'P3':(-0.4,-0.2),
    'Pz':(0,-0.25),'P4':(0.4,-0.2),'P8':(0.9,-0.2),'POz':(0,-0.5),'O1':(-0.4,-0.7),'Oz':(0,-0.7),'O2':(0.4,-0.7),
    'AF3':(-0.25,0.8),'AF4':(0.25,0.8)
}

# -------------------------- H‑PLI topography --------------------------

def hpli_topography(RECORDS, sr_channel, harmonics=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
                    time_col='Timestamp', half_bw=0.6, win_sec=8.0, step_sec=1.0,
                    n_perm=200, windows=None, out_dir='exports_spatial', show=True):
    """Compute per‑electrode H‑PLI_k maps + p‑values for each harmonic, and HCS map.
    windows: dict name->[(t0,t1),...] to restrict; if None uses all samples.
    Returns a dict with maps and saves topography PNGs.
    """
    ensure_dir(out_dir)
    ensure_timestamp_column(RECORDS, time_col=time_col)
    fs = infer_fs(RECORDS, time_col)

    EEGS = detect_eeg_channels(RECORDS)
    if sr_channel not in RECORDS.columns:
        raise ValueError(f"{sr_channel} not in dataframe columns")
    x_sr = pd.to_numeric(RECORDS[sr_channel], errors='coerce').fillna(0.0).values.astype(float)

    # windows to samples
    def wins_to_seg(wins, N):
        if not wins: return [(0,N)]
        segs=[]
        for (a,b) in wins:
            i0,i1=int(round(a*fs)), int(round(b*fs))
            i0=max(0,i0); i1=min(N,i1)
            if i1>i0: segs.append((i0,i1))
        return segs

    segs = wins_to_seg((None if windows is None else sum(windows.values(), [])), len(RECORDS))

    # phases for SR per harmonic (full length)
    PHI_SR = {f0: phase_series(x_sr, fs, f0, half_bw) for f0 in harmonics if f0 <= min(60.0, 0.999*0.5*fs)}

    HPLI = {f0:{} for f0 in PHI_SR}
    PV   = {f0:{} for f0 in PHI_SR}

    rng = np.random.default_rng(7)
    win = int(round(win_sec*fs)); step=int(round(step_sec*fs))
    centers = np.arange(win//2, len(RECORDS)-win//2, step, dtype=int)

    for ch in EEGS:
        x = pd.to_numeric(RECORDS[ch], errors='coerce').fillna(0.0).values.astype(float)
        for f0, phi_sr in PHI_SR.items():
            phi_e = phase_series(x, fs, f0, half_bw)
            # restrict to segs
            m = np.zeros_like(phi_e, dtype=bool)
            for (i0,i1) in segs: m[i0:i1] = True
            idx_centers = [c for c in centers if m[c]]
            vals=[]
            for c in idx_centers:
                sl = slice(c - win//2, c + win//2)
                dphi = phi_e[sl] - phi_sr[sl]
                vals.append(np.abs(np.mean(np.exp(1j*dphi))))
            vals = np.asarray(vals, float)
            hval = float(np.nanmean(vals)) if vals.size else np.nan
            # surrogate p: circular shift SR phase
            null=[]
            for _ in range(int(n_perm)):
                s = int(rng.integers(win, len(phi_sr)-1))
                phi_s = np.r_[phi_sr[-s:], phi_sr[:-s]]
                v=[]
                for c in idx_centers:
                    sl=slice(c-win//2, c+win//2)
                    d=phi_e[sl]-phi_s[sl]
                    v.append(np.abs(np.mean(np.exp(1j * d))))
#                     v.append(np.abs(np.mean(np.exp(1j,d))))
                null.append(np.nanmean(v) if v else np.nan)
            null = np.asarray(null, float)
            p = float((np.sum(null >= hval) + 1) / (np.sum(np.isfinite(null)) + 1)) if np.isfinite(hval) else np.nan
            HPLI[f0][ch] = hval; PV[f0][ch] = p

    # HCS per electrode (weights 1/k)
    weights = {}
    for f0 in PHI_SR.keys():
        k = max(1, int(round(f0/7.83)))
        weights[f0] = 1.0/float(k)
    wsum = sum(weights.values())
    HCS = {}
    for ch in EEGS:
        s=0.0; cnt=0
        for f0,vmap in HPLI.items():
            val = vmap.get(ch, np.nan)
            if np.isfinite(val):
                s += weights[f0]*val; cnt+=weights[f0]
        HCS[ch] = (s/(cnt+1e-12)) if cnt>0 else np.nan

    # GFS per harmonic (spatial phase consensus)
    GFS={}  # mean R across windows
    for f0 in PHI_SR:
        PHI_all=[]
        for ch in EEGS:
            x = pd.to_numeric(RECORDS[ch], errors='coerce').fillna(0.0).values.astype(float)
            PHI_all.append(phase_series(x, fs, f0, half_bw))
        PHI_all = np.vstack(PHI_all)  # (C,N)
        # restrict to segs
        m = np.zeros(PHI_all.shape[1], dtype=bool)
        for (i0,i1) in segs: m[i0:i1] = True
        R = np.abs(np.mean(np.exp(1j*PHI_all[:, m]), axis=0)) if np.any(m) else np.array([])
        GFS[f0] = float(np.nanmean(R)) if R.size else np.nan

    # -------- topography plotting --------
    try:
        import mne
        HAS_MNE = True
    except Exception:
        HAS_MNE = False

    def _plot_topo(values_dict, title, fname):
        chs=list(values_dict.keys()); vals=np.array([values_dict[c] for c in chs], float)
        if HAS_MNE:
            fs_local = fs
            info = mne.create_info(chs, sfreq=fs_local, ch_types=['eeg']*len(chs))
            montage = mne.channels.make_standard_montage('standard_1020')
            try:
                info.set_montage(montage, match_case=False)
                pos = np.array([info.get_montage().get_positions()['ch_pos'][c] for c in chs])[:, :2]
            except Exception:
                # fallback to dict
                pos = np.array([_COORDS_2D.get(c.replace('EEG.',''), (np.nan,np.nan)) for c in chs])
        else:
            pos = np.array([_COORDS_2D.get(c.replace('EEG.',''), (np.nan,np.nan)) for c in chs])
        # drop NaN positions
        mask = np.isfinite(pos).all(axis=1)
        chs = [c for c,m in zip(chs,mask) if m]
        vals = vals[mask]; pos = pos[mask]
        plt.figure(figsize=(5.2,4.6))
        sc = plt.scatter(pos[:,0], pos[:,1], c=vals, s=220, cmap='viridis', vmin=np.nanpercentile(vals,5), vmax=np.nanpercentile(vals,95))
        plt.colorbar(sc,label='value');
        for (x,y,c) in zip(pos[:,0], pos[:,1], chs):
            plt.text(x,y,c.replace('EEG.',''), ha='center', va='center', fontsize=6, color='k')
        plt.title(title); plt.axis('off'); plt.tight_layout();
        plt.savefig(os.path.join(out_dir,fname), dpi=160)
        plt.show(); plt.close()

    for f0 in PHI_SR:
        _plot_topo({ch:HPLI[f0].get(ch,np.nan) for ch in EEGS}, title=f'H‑PLI topography @ {f0:.2f} Hz', fname=f'hpli_topo_{f0:.2f}Hz.png')
    _plot_topo(HCS, title='HCS (weighted sum of H‑PLI across orders)', fname='hcs_topo.png')

    # GFS bar
    fig, ax = plt.subplots(figsize=(7,3))
    f_list = list(PHI_SR.keys()); gvals=[GFS[f] for f in f_list]
    ax.bar(np.arange(len(f_list)), gvals, width=0.7)
    ax.set_xticks(np.arange(len(f_list))); ax.set_xticklabels([f'{f:.1f}' for f in f_list])
    ax.set_ylabel('GFS (mean resultant across channels)'); ax.set_title('Global Field Synchronization per harmonic')
    ax.grid(True, axis='y', alpha=0.25, linestyle=':')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'gfs_bar.png'), dpi=160)
    plt.show(); plt.close()

    return {'HPLI':HPLI, 'PV':PV, 'HCS':HCS, 'GFS':GFS, 'EEG_channels':EEGS, 'fs':fs, 'out_dir':out_dir}

# -------------------------- PLV networks per order --------------------------

def plv_networks(RECORDS, channels=None, harmonics=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
                  time_col='Timestamp', half_bw=0.6, windows=None, thr_pct=0.2,
                  out_dir='exports_spatial', show=True):
    """Compute PLV networks at each harmonic; save adjacency heatmaps & graph plots; return stats (modularity, min‑cut, path length).
    channels: list of EEG.* columns to include; if None uses all EEG.* channels present.
    thr_pct: keep top p fraction of edges (0..1) for graph metrics.
    """
    ensure_dir(out_dir)
    ensure_timestamp_column(RECORDS, time_col=time_col)
    fs = infer_fs(RECORDS, time_col)
    if channels is None:
        channels = detect_eeg_channels(RECORDS)
    Nch = len(channels)

    # windows mask
    def wins_mask(N):
        if not windows: return np.ones(N, dtype=bool)
        m = np.zeros(N, dtype=bool)
        for wins in windows.values():
            for (a,b) in wins:
                i0,i1=int(round(a*fs)), int(round(b*fs))
                i0=max(0,i0); i1=min(N,i1)
                m[i0:i1]=True
        return m

    mask = wins_mask(len(RECORDS))

    # precompute phases for all channels x harmonics
    X = {ch: pd.to_numeric(RECORDS[ch], errors='coerce').fillna(0.0).values.astype(float) for ch in channels}
    PHI = {f0: {ch: phase_series(X[ch], fs, f0, half_bw) for ch in channels if f0 <= min(60.0, 0.999*0.5*fs)} for f0 in harmonics}

    stats_rows=[]

    for f0 in PHI:
        # PLV matrix
        A = np.zeros((Nch,Nch), float)
        for i,ch_i in enumerate(channels):
            ph_i = PHI[f0][ch_i][mask]
            for j,ch_j in enumerate(channels[i+1:], start=i+1):
                ph_j = PHI[f0][ch_j][mask]
                L = min(len(ph_i), len(ph_j))
                if L<100:
                    val = np.nan
                else:
                    val = float(np.abs(np.mean(np.exp(1j*(ph_i[:L]-ph_j[:L])))))
                A[i,j]=A[j,i]=val
        # heatmap
        plt.figure(figsize=(5.8,5.0))
        v = np.nan_to_num(A, nan=0.0)
        plt.imshow(v, origin='lower', cmap='viridis', vmin=0, vmax=1)
        plt.xticks(range(Nch), [c.replace('EEG.','') for c in channels], rotation=90, fontsize=7)
        plt.yticks(range(Nch), [c.replace('EEG.','') for c in channels], fontsize=7)
        plt.colorbar(label='PLV'); plt.title(f'PLV matrix @ {f0:.2f} Hz'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'plv_matrix_{f0:.2f}Hz.png'), dpi=160)
        if show: plt.show(); plt.close()

        # threshold top p% edges (exclude diagonal)
        triu = A[np.triu_indices(Nch,1)]
        finite = triu[np.isfinite(triu)]
        if finite.size == 0:
            stats_rows.append({'f0':f0, 'modularity':np.nan, 'min_cut':np.nan, 'avg_path_len':np.nan, 'density':np.nan})
            continue
        thresh = np.nanpercentile(finite, 100*(1.0-thr_pct))
        G = nx.Graph()
        G.add_nodes_from(range(Nch))
        for i in range(Nch):
            for j in range(i+1,Nch):
                w = A[i,j]
                if np.isfinite(w) and w >= thresh:
                    G.add_edge(i,j,weight=float(w))
        # stats
        density = nx.density(G)
        # communities & modularity (falls back to NaN if <2 communities)
        try:
            comm = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
            modules = [set(c) for c in comm]
            modularity = nx.algorithms.community.modularity(G, modules, weight='weight') if len(modules)>=2 else np.nan
        except Exception:
            modularity = np.nan
        # min‑cut
        try:
            cut_val, part = nx.algorithms.connectivity.stoer_wagner(G, weight='weight')
            min_cut = float(cut_val)
        except Exception:
            min_cut = np.nan
        # path length on largest component
        if len(G) == 0 or not nx.is_connected(G.to_undirected(as_view=True)):
            comps = list(nx.connected_components(G))
            if comps:
                H = G.subgraph(max(comps, key=len)).copy()
            else:
                H = None
        else:
            H = G
        try:
            avg_pl = nx.average_shortest_path_length(H, weight=None) if H and H.number_of_edges()>0 else np.nan
        except Exception:
            avg_pl = np.nan
        stats_rows.append({'f0':f0, 'modularity':modularity, 'min_cut':min_cut, 'avg_path_len':avg_pl, 'density':density})

        # simple graph plot
        pos = nx.circular_layout(G)
        plt.figure(figsize=(6,5))
        nx.draw_networkx_nodes(G, pos, node_size=160)
        # scale edge width by weight
        widths=[2.0*G[u][v]['weight'] for u,v in G.edges()] if G.number_of_edges()>0 else []
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.7)
        nx.draw_networkx_labels(G, pos, labels={i:channels[i].replace('EEG.','') for i in G.nodes()}, font_size=8)
        plt.title(f'PLV graph (top {int(thr_pct*100)}% edges) @ {f0:.2f} Hz')
        plt.axis('off'); plt.tight_layout();
        plt.savefig(os.path.join(out_dir, f'plv_graph_{f0:.2f}Hz.png'), dpi=160)
        if show: plt.show(); plt.close()

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(os.path.join(out_dir, 'plv_network_stats.csv'), index=False)
    return stats

# -------------------------- Source loc & connectome overlap --------------------------

def lcmv_sources_at_lines(RECORDS, eeg_channels, sr=None, fwd=None, noise_cov=None,
                           time_col='Timestamp', lines=(7.83,14.3,20.8,27.3,33.8),
                           half_bw=0.6, windows=None, out_dir='exports_spatial', show=True):
    """If MNE forward model & noise covariance are provided, compute LCMV source power maps
    at each SR line; else returns None. Saves one figure per line.
    """
    try:
        import mne
    except Exception:
        print('[lcmv] MNE not available; skipping source localization.')
        return None
    if fwd is None or noise_cov is None:
        print('[lcmv] Forward model or noise covariance missing; skipping source localization.')
        return None

    ensure_dir(out_dir)
    ensure_timestamp_column(RECORDS, time_col=time_col)
    fs = infer_fs(RECORDS, time_col)

    # build mne.Raw from dataframe columns
    info = mne.create_info(eeg_channels, sfreq=fs, ch_types=['eeg']*len(eeg_channels))
    montage = mne.channels.make_standard_montage('standard_1020')
    try:
        info.set_montage(montage, match_case=False)
    except Exception:
        pass
    data = np.vstack([pd.to_numeric(RECORDS[ch], errors='coerce').fillna(0.0).values.astype(float) for ch in eeg_channels])
    raw = mne.io.RawArray(data, info)

    # windows mask
    if windows:
        m = np.zeros(raw.n_times, dtype=bool)
        for wins in windows.values():
            for (a,b) in wins:
                i0,i1=int(round(a*fs)), int(round(b*fs))
                m[i0:i1]=True
        raw = raw.copy().load_data()
        raw._data[:, ~m] = 0.0

    src_maps = {}
    for f0 in lines:
        if f0 > min(60.0, 0.999*0.5*fs):
            continue
        l_freq = max(0.01, f0-half_bw); h_freq = f0+half_bw
        raw_f = raw.copy().filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        data_cov = mne.compute_raw_covariance(raw_f, method='oas', verbose=False)
        filters = mne.beamformer.make_lcmv(raw_f.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                           pick_ori='max-power', weight_norm='unit-noise-gain', verbose=False)
        stc = mne.beamformer.apply_lcmv_raw(raw_f, filters, max_ori_out='signed', verbose=False)
        src_maps[f0] = stc
        # simple brain plot (if fsaverage/subjects_dir available)
        try:
            brain = stc.plot(hemi='split', views=['lat'], time_viewer=False, smoothing_steps=5,
                             clim='auto', colormap='magma')
            brain.save_image(os.path.join(out_dir, f'lcmv_{f0:.2f}Hz.png'))
            brain.close()
        except Exception:
            pass
    return src_maps

# Connectome harmonic overlap

def connectome_overlap(source_vec, SC, k_modes=10):
    """Project a source map (N nodes) onto Laplacian eigenmodes of SC; return variance explained per mode."""
    import numpy as np
    # Laplacian
    D = np.diag(SC.sum(1))
    L = D - SC
    w, V = np.linalg.eigh(L)
    # sort by ascending eigenvalue (mode‑1 = global/slowest)
    idx = np.argsort(w); w=w[idx]; V=V[:,idx]
    x = source_vec - np.mean(source_vec)
    coeff = V.T @ x
    var = coeff**2
    frac = var/np.sum(var+1e-12)
    return {'eigs':w, 'frac':frac, 'V':V, 'coeff':coeff, 'topk':(w[:k_modes], frac[:k_modes])}

# -------------------------- Example orchestrator --------------------------

def analyze_spatial_and_source(RECORDS,
                               sr_channel,
                               windows,
                               harmonics=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
                               half_bw=0.6,
                               out_dir='exports_spatial',
                               show=True,
                               do_networks=True,
                               network_channels=None,
                               do_sources=False,
                               mne_fwd=None, mne_noise_cov=None,
                               do_connectome=False,
                               SC=None):
    """Run spatial topographies (H‑PLI, HCS), GFS bars, (optional) PLV networks and LCMV sources.
    Optional connectome overlap if SC is provided (N×N). Returns a dict of outputs.
    """
    ensure_dir(out_dir)
    topo = hpli_topography(RECORDS, sr_channel, harmonics=harmonics, half_bw=half_bw,
                           windows=windows, out_dir=out_dir, show=show)
    out = {'topography': topo}

    if do_networks:
        net_stats = plv_networks(RECORDS, channels=network_channels, harmonics=harmonics,
                                 half_bw=half_bw, windows=windows, out_dir=out_dir, show=show)
        out['network_stats'] = net_stats

    if do_sources:
        eeg_chs = detect_eeg_channels(RECORDS)
        stcs = lcmv_sources_at_lines(RECORDS, eeg_chs, fwd=mne_fwd, noise_cov=mne_noise_cov,
                                     lines=tuple([f for f in harmonics if f<=60.0]),
                                     half_bw=half_bw, windows=windows, out_dir=out_dir, show=show)
        out['sources'] = stcs
        # connectome overlap if provided and stc is vector in SC space
        if do_connectome and (SC is not None) and (stcs is not None):
            # This requires mapping stc to SC node space; placeholder expects `source_vec` aligned to SC
            # Example: take absolute mean across time for each source vertex and provide a vertex→SC mapping externally.
            pass

    return out

# -------------------------- Quick usage --------------------------
# windows = {
#   'baseline': [(0, 290)],
#   'ignition': [(290, 310), (580, 600)],
#   'rebound':  [(325, 580)]
# }
# out = analyze_spatial_and_source(
#     RECORDS,
#     sr_channel='EEG.Pz',   # or your magnetometer channel
#     windows=windows,
#     harmonics=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
#     half_bw=0.6,
#     out_dir='exports_spatial',
#     show=True,
#     do_networks=True,
#     network_channels=None,   # None = all EEG.* channels
#     do_sources=False,        # set True if you provide MNE forward model & noise_cov
#     mne_fwd=None, mne_noise_cov=None,
#     do_connectome=False,
#     SC=None
# )
