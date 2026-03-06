"""Session Metadata Parsing Utility.

Extracts subject, device, context, and dataset identifiers from EEG session filenames.
Each dataset has specific filename patterns that encode metadata differently.
"""

import os


def parse_session_metadata(filepath, dataset_name):
    """Parse metadata from filepath and dataset name.

    Parameters
    ----------
    filepath : str
        Full or partial path to the EEG session file.
    dataset_name : str
        Dataset identifier: 'files', 'physf', 'vep', 'mpeng1', 'mpeng2', 'insight', 'muse'.

    Returns
    -------
    dict
        Metadata dictionary with keys:
        - subject: Subject identifier (e.g., 'm1', 's10', 'sub10', '314')
        - device: EEG device ('epoc', 'insight', 'muse')
        - context: Recording context ('meditation', 'flow', 'visual', 'gaming')
        - dataset: Dataset name (normalized lowercase)

    Examples
    --------
    >>> parse_session_metadata('data/PhySF/s10_flow.csv', 'physf')
    {'subject': 's10', 'device': 'epoc', 'context': 'flow', 'dataset': 'physf'}

    >>> parse_session_metadata('data/vep/sub10_A2.csv', 'vep')
    {'subject': 'sub10', 'device': 'epoc', 'context': 'visual', 'dataset': 'vep'}
    """
    filename = os.path.basename(filepath)
    name_no_ext = filename.replace('.csv', '')
    dataset_lower = dataset_name.lower()

    # Default values
    metadata = {
        'subject': 'unknown',
        'device': 'epoc',
        'context': 'unknown',
        'dataset': dataset_lower
    }

    # Normalize mpeng1/mpeng2 to 'mpeng' for dataset
    if dataset_lower in ('mpeng1', 'mpeng2'):
        metadata['dataset'] = 'mpeng'

    # Parse based on dataset
    if dataset_lower == 'files':
        # FILES: subject='m1', context='meditation', device from filename
        # Examples:
        #   "20201229_29.12.20_11.27.57.md.pm.bp.csv"
        #   "med_EPOCX_111270_2021.06.12T09.50.52.04.00.md.bp.csv"
        #   "binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv"
        #   "test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv"
        metadata['subject'] = 'm1'
        metadata['context'] = 'meditation'

        # Infer device from filename
        fn_upper = filename.upper()
        if 'EPOC' in fn_upper:
            metadata['device'] = 'epoc'
        elif 'INSIGHT' in fn_upper:
            metadata['device'] = 'insight'
        elif 'MUSE' in fn_upper:
            metadata['device'] = 'muse'
        else:
            # Default to epoc for this dataset (all confirmed EPOCX)
            metadata['device'] = 'epoc'

    elif dataset_lower == 'physf':
        # PHYSF: pattern "<subject>_<context>.csv"
        # Examples: "s10_flow.csv", "s10_no_flow.csv", "s11_no_flow.csv"
        metadata['device'] = 'epoc'

        # Split on first underscore only to handle "no_flow" contexts
        parts = name_no_ext.split('_', 1)
        if len(parts) >= 1:
            metadata['subject'] = parts[0]
        if len(parts) >= 2:
            metadata['context'] = parts[1]

    elif dataset_lower == 'vep':
        # VEP: pattern "<subject>_<condition>.csv", context='visual'
        # Examples: "sub10_A2.csv", "sub11_C1.csv", "sub12_P2.csv"
        metadata['device'] = 'epoc'
        metadata['context'] = 'visual'

        parts = name_no_ext.split('_')
        if len(parts) >= 1:
            metadata['subject'] = parts[0]

    elif dataset_lower in ('mpeng', 'mpeng1', 'mpeng2'):
        # MPENG: pattern "<subject>_<session_params>..."
        # Examples: "314_383_4_4_2_4 2.csv", "314_384_4_3_3_4 2.csv"
        metadata['device'] = 'epoc'
        metadata['context'] = 'gaming'

        parts = name_no_ext.split('_')
        if len(parts) >= 1:
            metadata['subject'] = parts[0]

    elif dataset_lower == 'insight':
        # INSIGHT: Emotiv Insight device
        metadata['device'] = 'insight'
        metadata['context'] = 'meditation'
        metadata['subject'] = 'm1'

    elif dataset_lower == 'muse':
        # MUSE: Muse headband device
        metadata['device'] = 'muse'
        metadata['context'] = 'meditation'
        metadata['subject'] = 'm1'

    elif dataset_lower == 'arithmetic':
        # Arithmetic EEG: NeXus-32, pattern "S<nn><condition>.csv"
        # Experiment 1: S01–S21, conditions A/M/R
        # Experiment 2: S10–S41, conditions A/B/R
        metadata['device'] = 'nexus32'
        condition_map = {
            'A': 'arithmetic',
            'M': 'meditation',
            'B': 'baseline',
            'R': 'rest',
        }
        if len(name_no_ext) >= 4 and name_no_ext[0].upper() == 'S':
            metadata['subject'] = name_no_ext[:3]  # e.g. "S01"
            cond_code = name_no_ext[3:]             # e.g. "A"
            metadata['context'] = condition_map.get(cond_code, cond_code)

    # Prepend dataset name to subject for unique IDs across datasets
    if metadata['subject'] != 'unknown':
        metadata['subject'] = f"{metadata['dataset']}_{metadata['subject']}"

    return metadata


def add_metadata_to_results(results_list, filepath, dataset_name):
    """Add metadata columns to a list of result dictionaries.

    Parameters
    ----------
    results_list : list of dict
        List of result dictionaries (e.g., from detect_ignition).
    filepath : str
        Path to the EEG session file.
    dataset_name : str
        Dataset identifier.

    Returns
    -------
    list of dict
        Results with added metadata columns.
    """
    metadata = parse_session_metadata(filepath, dataset_name)

    for result in results_list:
        result.update(metadata)

    return results_list


def get_dataset_config():
    """Return standard dataset configurations for reference.

    Returns
    -------
    dict
        Dataset configurations with default metadata values.
    """
    return {
        'files': {
            'default_subject': 'm1',
            'default_device': 'epoc',
            'default_context': 'meditation',
            'header': 1,
            'electrodes': 'epoc'
        },
        'physf': {
            'default_device': 'epoc',
            'header': 0,
            'electrodes': 'epoc',
            'pattern': '<subject>_<context>.csv'
        },
        'vep': {
            'default_device': 'epoc',
            'default_context': 'visual',
            'header': 0,
            'electrodes': 'epoc',
            'pattern': '<subject>_<condition>.csv'
        },
        'mpeng1': {
            'default_device': 'epoc',
            'default_context': 'gaming',
            'header': 0,
            'electrodes': 'epoc',
            'pattern': '<subject>_<params>...'
        },
        'mpeng2': {
            'default_device': 'epoc',
            'default_context': 'gaming',
            'header': 0,
            'electrodes': 'epoc',
            'pattern': '<subject>_<params>...'
        },
        'insight': {
            'default_subject': 'm1',
            'default_device': 'insight',
            'default_context': 'meditation',
            'header': 1,
            'electrodes': 'insight'
        },
        'muse': {
            'default_device': 'muse',
            'default_context': 'meditation',
            'header': 1,
            'electrodes': 'muse',
            'pattern': '<device_id>_<timestamp>_<unix_ts>.csv'
        },
        'arithmetic': {
            'default_device': 'nexus32',
            'header': 0,
            'electrodes': 'nexus32',
            'pattern': 'S<subject><condition>.csv',
            'conditions': {'A': 'arithmetic', 'M': 'meditation', 'B': 'baseline', 'R': 'rest'},
            'fs': 256
        }
    }
