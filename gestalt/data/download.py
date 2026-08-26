# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download the source datasets."""
import argparse
import os
import subprocess
import tarfile

HEST_HF_REPO = 'MahmoodLab/hest'

COHORTS = {
    'hest-lung': [f'NCBI{i}' for i in (856, 857, 858, 859, 860, 861, 864, 865, 866, 867,
                                       870, 873, 875, 880, 881, 882)],
    'hest-breast': [f'TENX{i}' for i in range(191, 203)],
}

ZENODO_RECORD = '17429142'
ZENODO_FILES = f'https://zenodo.org/records/{ZENODO_RECORD}/files'

P1_CELLS_PARQUET_URL = ('https://cf.10xgenomics.com/samples/xenium/3.0.0/'
                         'Xenium_Prime_Human_Skin_FFPE/Xenium_Prime_Human_Skin_FFPE_cells.parquet')
XENIUM = {
    'C1': dict(name='Xenium_V1_hSkin_Melanoma_Base_FFPE', size_gb=32.0),
    'C2': dict(name='Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE', size_gb=26.0),
    'P1': dict(name='Xenium_Prime_Human_Skin_FFPE', size_gb=56.3, needs_cells_parquet=True),
}


def fetch_hest(samples, out):
    from huggingface_hub import snapshot_download
    if not os.environ.get('HF_TOKEN'):
        raise SystemExit('HEST-1k is gated. Accept the terms on the HuggingFace model card and '
                         'export HF_TOKEN before running this.')
    os.makedirs(out, exist_ok=True)
    patterns = [f'*{s}*' for s in samples]
    snapshot_download(repo_id=HEST_HF_REPO, repo_type='dataset', local_dir=out,
                      allow_patterns=patterns)
    print(f'fetched {len(samples)} HEST samples -> {out}')


def fetch_xenium(out, slides=None):
    """Download and extract the requested Xenium slides."""
    os.makedirs(out, exist_ok=True)
    for k in (slides or list(XENIUM)):
        v = XENIUM[k]
        name = v['name']
        tar_path = f'{out}/{name}.tar.gz'
        extracted_dir = f'{out}/{name}'
        if os.path.isdir(extracted_dir):
            print(f'{k}: {extracted_dir} already extracted, skipping')
        else:
            if not os.path.exists(tar_path):
                print(f'{k}: downloading {name}.tar.gz (~{v["size_gb"]:.1f}GB) from Zenodo...')
            else:
                print(f'{k}: resuming partial download of {name}.tar.gz...')
            url = f'{ZENODO_FILES}/{name}.tar.gz?download=1'
            subprocess.run(['wget', '-c', '-O', tar_path, url], check=True)
            print(f'{k}: extracting...')
            with tarfile.open(tar_path) as tf:
                tf.extractall(out)
            print(f'{k}: done -> {extracted_dir}')
        if v.get('needs_cells_parquet'):
            parquet_path = f'{out}/{name}_cells.parquet'
            if os.path.exists(parquet_path):
                print(f'{k}: {parquet_path} already present, skipping')
            else:
                print(f'{k}: downloading cells.parquet...')
                subprocess.run(['wget', '-c', '-O', parquet_path, P1_CELLS_PARQUET_URL], check=True)
            print(f'{k}: pass --coords-parquet {parquet_path} to gestalt.data.tiles')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cohort', required=True, choices=list(COHORTS) + ['xenium'])
    ap.add_argument('--out', default=os.environ.get('GESTALT_RAW', './data/raw'))
    ap.add_argument('--slides', nargs='+', choices=list(XENIUM), default=None,
                    help='xenium cohort only: subset of C1/C2/P1 to fetch (default: all three)')
    a = ap.parse_args()
    if a.cohort == 'xenium':
        fetch_xenium(a.out, a.slides)
    else:
        fetch_hest(COHORTS[a.cohort], a.out)


if __name__ == '__main__':
    main()
