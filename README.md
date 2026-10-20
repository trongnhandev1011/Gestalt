# Gestalt

Official implementation of Gestalt for histopathology image synthesis from spatial gene
expression.

## Installation

```bash
pip install -r requirements.txt
```

A CUDA GPU and a Hugging Face token are required. Configure the data and checkpoint paths:

```bash
export HF_TOKEN=...
export GESTALT_RAW=/path/to/raw
export SPATIALGENE_OUT=/path/to/tiles
export SPATIALGENE_CKPT=/path/to/checkpoints
```

## Data preparation

Download the required cohorts:

```bash
python -m gestalt.data.download --cohort hest-lung
python -m gestalt.data.download --cohort hest-breast
python -m gestalt.data.download --cohort xenium
```

Prepare selected HEST samples:

```bash
python -m gestalt.data.preprocess_hest \
    --samples NCBI856,NCBI857,NCBI858 \
    --root "$GESTALT_RAW" \
    --out "$SPATIALGENE_OUT"
```

Prepare Xenium C1, C2, and P1 at native 40x resolution:

```bash
python -m gestalt.data.tiles \
    --slides C1 C2 \
    --layout overlap \
    --target-umpp 0.25 \
    --root "$GESTALT_RAW"

python -m gestalt.data.tiles \
    --slides P1 \
    --layout overlap \
    --target-umpp 0.25 \
    --root "$GESTALT_RAW" \
    --coords-parquet /path/to/Xenium_Prime_Human_Skin_FFPE_cells.parquet
```

For the single-cell protocol:

```bash
python -m gestalt.data.tiles \
    --slides C1 C2 P1 \
    --layout singlecell \
    --target-umpp 0.25 \
    --root "$GESTALT_RAW"
```

UNI2-h embeddings are extracted during tiling. To use another foundation model:

```bash
python -m gestalt.data.extract_fm --slides C1 C2 P1 --fm conch
```

Supported values of `--fm` are `uni2h`, `uni_v1`, `conch`, and `virchow2`.

## Train the mapper

```bash
LUNG=NCBI856,NCBI857,NCBI858,NCBI859,NCBI860,NCBI861,NCBI864,NCBI865,NCBI866,NCBI867,NCBI870,NCBI873,NCBI875,NCBI880,NCBI881,NCBI882

python -m gestalt.tile256.train_mapper \
    --samples "$LUNG" \
    --test-sample NCBI875 \
    --fm uni2h \
    --pairbias-k 8 \
    --split-seed 0 \
    --seed 42 \
    --snapshot-every 4
```

## Adapt the decoder

```bash
python -m gestalt.tile256.lora \
    --signal mapper \
    --samples "$LUNG" \
    --fm uni2h \
    --rank 32 \
    --alpha 64 \
    --steps 8000 \
    --save-every 8000 \
    --split-seed 0 \
    --train-seed 42
```

## Benchmark

```bash
python -m gestalt.tile256.benchmark \
    --full-sample \
    --samples "$LUNG" \
    --sample NCBI875 \
    --fm uni2h \
    --pairbias-k 8 \
    --split-seed 0 \
    --seed 42 \
    --sampling-seed 42 \
    --seed-check \
    --mapper-ckpt "$SPATIALGENE_CKPT/<mapper>.pt" \
    --pixcell-lora "$SPATIALGENE_CKPT/<adapter>/step_8000"
```

For the reported three-seed evaluation, use seeds `7`, `42`, and `13`. Keep
`--split-seed 0` fixed and use the run seed for the other seed arguments.

## 1024px region synthesis

```bash
python -m gestalt.region1024.train \
    --samples C1 \
    --fm uni2h \
    --readout-decay 50

python -m gestalt.region1024.benchmark \
    --samples C1 \
    --sample C1
```

## License

See [LICENSE](LICENSE).
