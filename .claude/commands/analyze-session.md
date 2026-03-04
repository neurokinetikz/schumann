Analyze the EEG session output and generate a comprehensive summary report.

Session to analyze: $ARGUMENTS

## Instructions

1. **Locate Session Files**
   - Find the session output directory in `exports/`
   - Identify all 7 PNG charts and the markdown file

2. **Read All Output Files**
   - Read each PNG chart using the Read tool (they will be displayed visually)
   - Read the markdown console output file for numerical metrics

3. **Analyze Each Chart**

   For each of the 7 charts, provide detailed interpretation:

   **1. Raw EEG Signal (_1_raw_eeg.png)**
   - Overall signal quality and amplitude characteristics
   - Visible artifacts or anomalies
   - General activity patterns

   **2. F0 Filtered Signal (_2_f0_filtered.png)**
   - Schumann fundamental (~7.6 Hz) band activity
   - Amplitude modulations and patterns
   - Correlation with ignition events

   **3. Z-Scored Envelope (_3_z_envelope.png)**
   - Detection threshold crossings (z > 3.0)
   - Ignition event timing and duration
   - False positive/negative assessment

   **4. Mean Squared Coherence (_4_msc.png)**
   - MSC values during ignition vs baseline
   - Temporal coherence patterns
   - Evidence of inter-electrode synchronization

   **5. Phase Locking Value (_5_plv.png)**
   - PLV during ignition windows
   - Phase stability assessment
   - Comparison to baseline periods

   **6. Bicoherence Triads (_6_bicoherence.png)**
   - Cross-frequency coupling patterns
   - f0-f0, f0-f1, f1-f1, f0-f1-f2 triad activity
   - Ignition vs baseline differences

   **7. Effect Sizes (_7_effect_sizes.png)**
   - Cohen's d for each metric
   - Statistical significance indicators
   - Overall effect magnitude assessment

4. **Extract Key Metrics from Markdown**
   - Session duration and event count
   - Detected harmonics (f0, f1, f2 frequencies)
   - Effect sizes with p-values
   - Coherence statistics
   - FOOOF parameters

5. **Generate Summary Report**

   Create a comprehensive markdown report with:

   ```markdown
   # Session Summary Report: {session_name}

   ## Overview
   - Duration, events, time in ignition state
   - Signal quality assessment
   - Key findings summary

   ## Chart Analysis
   [Detailed analysis of each chart]

   ## Key Metrics
   | Metric | Ignition | Baseline | Effect Size | p-value |
   |--------|----------|----------|-------------|---------|

   ## Session Assessment
   - Evidence strength for brain-SR coupling
   - Data quality rating
   - Confidence in findings

   ## Notable Patterns
   - Unusual features
   - Interesting observations

   ## Recommendations
   - Follow-up analyses
   - Caveats to consider
   ```

6. **Save Report**
   - Save as `{session_name}_summary_report.md` in the session directory
