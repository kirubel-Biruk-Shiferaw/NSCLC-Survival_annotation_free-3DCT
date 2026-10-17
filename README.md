# NSCLC Survival Prediction — Reproducible Deep Learning Pipeline

[![CI](https://github.com/kirubel-Biruk-Shiferaw/NSCLC-Survival_annotation_free-3DCT/actions/workflows/ci.yml/badge.svg)](https://github.com/kirubel-Biruk-Shiferaw/NSCLC-Survival_annotation_free-3DCT/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Reproducible code for 17 deep-learning survival analysis experiments (E1–E17d)
on the TCIA **NSCLC-Radiomics / Lung1 cohort** (n=422).  
Models predict overall survival from 3-D CT images and clinical variables
using Cox partial likelihood (Harrell's C-index evaluation).

---

## COLIPRI Pretraining Data

Experiments E16a/b and E17a/b/c/d use `microsoft/colipri` embeddings.
COLIPRI (Wald et al., ECCV 2026) is pretrained on **CT-RATE** (25,692 chest
CT volumes paired with radiology reports) and **NLST** (~72,000 image-only
chest CT scans) — see the paper's dataset table. **Neither corpus includes the
NSCLC-Radiomics / Lung1 cohort used here**: Lung1 is from MAASTRO Clinic (TCIA),
CT-RATE is from Istanbul Medipol University Hospital, and NLST is the US
National Lung Screening Trial. The Lung1 cohort is therefore not part of
COLIPRI's pretraining data, so the COLIPRI results are not affected by that
form of train/test leakage. As with every single-cohort study, external
validation on an independent dataset is still required before clinical claims.
See [docs/model_card.md](docs/model_card.md).

---

## Repository Structure

```
nsclc_survival/              # Installable Python package
├── utils/
│   ├── seed.py              # set_seed(), SEED=123
│   └── config.py            # NSCLCConfig dataclass, load_config()
├── data/
│   ├── clinical.py          # load_labels(), normalise_age_per_fold()
│   ├── splits.py            # make_splits(), load_splits()
│   └── datasets.py          # DS_Base, DS_Clinical, DS_COLIPRI_Patch, …
├── losses/
│   └── cox.py               # cox_loss_breslow(), cox_loss_efron()
├── models/
│   ├── cnn.py               # E1–E15 architectures
│   └── colipri.py           # E16a/b, E17a/b/c/d architectures
├── training/
│   └── trainer.py           # run_experiment(), train_one_fold()
├── evaluation/
│   └── metrics.py           # bootstrap_cindex(), wilcoxon_compare(), IBS
└── interpretability/
    └── gradcam.py           # GradCAM3D, compute_atr()

scripts/
├── train.py                 # CLI: train all or specific experiments
├── evaluate.py              # CLI: compute metrics + CIs + Wilcoxon tests
└── interpret.py             # CLI: GradCAM + ATR

tests/
├── test_losses.py           # Cox loss correctness
├── test_splits.py           # CV split reproducibility
└── test_metrics.py          # C-index, Wilcoxon, IBS

configs/
└── default.yaml             # All hyperparameters + paths

docs/
└── model_card.md            # Architecture details, COLIPRI caveat, results
```

---

## Step-by-Step Reproduction Guide

### Step 1 — Clone and set up the environment

```bash
git clone https://github.com/kirubel-Biruk-Shiferaw/NSCLC-Survival_annotation_free-3DCT.git
cd NSCLC-Survival_annotation_free-3DCT

# Option A: conda (recommended for GPU support)
conda env create -f environment.yml
conda activate nsclc_survival

# Option B: pip
pip install -r requirements.txt
pip install -e .
```

### Step 2 — Download the TCIA Lung1 dataset

1. Visit the [TCIA NSCLC-Radiomics page](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/).
2. Accept the TCIA Data Usage Agreement.
3. Download via TCIA Data Retriever or `tcia_utils`:

```bash
pip install tcia_utils
python - <<'EOF'
from tcia_utils import nbia
nbia.downloadSeries(
    series_data=nbia.getSeries(collection="NSCLC-Radiomics"),
    path="data/raw/NSCLC-Radiomics",
)
EOF
```

4. Download the clinical CSV:
   - Direct link: https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics
   - File: `NSCLC Radiomics Lung1.clinical-v3a.csv`
   - Save to `data/NSCLC_Radiomics_Lung1.clinical.csv`

### Step 3 — Preprocess CT volumes to R2 format

R2 = resampled to 1×1×1 mm³, padded/cropped to (1, 64, 128, 128):

```bash
python - <<'EOF'
import nibabel as nib
import numpy as np
from skimage.transform import resize
from pathlib import Path

raw_dir = Path("data/raw/NSCLC-Radiomics")
r2_dir  = Path("data/R2")
r2_dir.mkdir(exist_ok=True)

VOL_SHAPE = (64, 128, 128)

for pid_dir in sorted(raw_dir.glob("LUNG1-*")):
    pid = pid_dir.name
    nifti = sorted(pid_dir.rglob("*.nii.gz"))[0]
    img = nib.load(str(nifti))
    vol = img.get_fdata()
    # Resample to VOL_SHAPE
    vol_r = resize(vol, VOL_SHAPE, order=1, anti_aliasing=True,
                   preserve_range=True)
    # Normalise HU to [0, 1]: clip [-1000, 400] → [0, 1]
    vol_r = np.clip(vol_r, -1000, 400)
    vol_r = (vol_r + 1000) / 1400.0
    out_path = r2_dir / f"{pid}.npy"
    np.save(str(out_path), vol_r.astype(np.float32))
    print(f"  {pid}: {vol_r.shape}  → {out_path}")
EOF
```

### Step 4 — Generate COLIPRI embeddings (for E16/E17 only)

Requires `microsoft/colipri` weights via HuggingFace (access may require approval):

```bash
pip install transformers>=4.40.0 huggingface_hub

python - <<'EOF'
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("microsoft/colipri").eval()
proc  = AutoProcessor.from_pretrained("microsoft/colipri")

r2_dir  = Path("data/R2")
emb_dir = Path("data/colipri_embeddings")
emb_dir.mkdir(exist_ok=True)

for npy_path in sorted(r2_dir.glob("*.npy")):
    pid = npy_path.stem
    vol = torch.from_numpy(np.load(str(npy_path))).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    with torch.no_grad():
        outputs = model(pixel_values=vol)
    # Pooled embedding (768-dim) and patch embeddings
    pooled  = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # (768,)
    patches = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy() # (N, 768)
    np.save(str(emb_dir / f"{pid}_pool.npy"),  pooled)
    np.save(str(emb_dir / f"{pid}_patch.npy"), patches)
    print(f"  {pid}: pool={pooled.shape}, patches={patches.shape}")
EOF
```

> **Note**: If COLIPRI access is unavailable, skip this step and train only
> experiments E1–E15 by passing `--exp E1` through `--exp E15`.

### Step 5 — Configure paths

Set environment variables (or edit `configs/default.yaml`):

```bash
export NSCLC_ARTIFACT_DIR="/path/to/data"        # contains R2/, CSV, colipri_embeddings/
export NSCLC_OUTPUT_DIR="/path/to/outputs"        # checkpoints, OOF predictions
export MEDICALNET_CKPT="/path/to/resnet_10.pth"  # optional: MedicalNet checkpoint
```

Or edit `configs/default.yaml` directly:

```yaml
data:
  artifact_dir: /path/to/data
  csv_filename: NSCLC_Radiomics_Lung1.clinical.csv

output:
  output_dir: /path/to/outputs
```

#### MedicalNet checkpoint (E14/E15 only)

Download from the official [MedicalNet repository](https://github.com/Tencent/MedicalNet):

```bash
# Download ResNet10 pretrained on 8 segmentation datasets
wget https://github.com/Tencent/MedicalNet/releases/download/pretraining/resnet_10_23dataset.pth \
     -O data/medicalnet/resnet_10.pth
export MEDICALNET_CKPT="data/medicalnet/resnet_10.pth"
```

### Step 6 — Run unit tests

```bash
pytest tests/ -v
```

Expected: all tests pass in ~30 seconds on CPU (no GPU or data required).

### Step 7 — Train

```bash
# Train all experiments (E1–E17d) — ~6 hours on one A100 GPU
python scripts/train.py --config configs/default.yaml

# Train a single experiment
python scripts/train.py --config configs/default.yaml --exp E2

# Train with MedicalNet
python scripts/train.py --config configs/default.yaml \
    --medicalnet-ckpt data/medicalnet/resnet_10.pth

# CNN-only experiments (skip COLIPRI)
for exp in E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E11 E12 E13 E14 E15; do
    python scripts/train.py --config configs/default.yaml --exp $exp
done
```

**Expected training times** (A100 40GB, batch_size=32):

| Experiments | Time |
|---|---|
| E1–E9 (5-fold CNN) | ~2.5 h |
| E10–E11 (clinical) | ~5 min |
| E12–E13 | ~45 min |
| E14–E15 (MedicalNet) | ~3 h |
| E16a–E17d (COLIPRI, 10-fold) | ~4 h |

### Step 8 — Evaluate

```bash
python scripts/evaluate.py --config configs/default.yaml

# Skip slow IBS computation:
python scripts/evaluate.py --config configs/default.yaml --no-ibs
```

Output files:
- `outputs/evaluation_results.json` — C-indices, bootstrap CIs, Wilcoxon p-values
- `outputs/results_table.md` — Markdown table ready for thesis/paper

### Step 9 — Interpretability (optional)

```bash
# GradCAM heatmaps for E3, fold 1
python scripts/interpret.py --config configs/default.yaml --exp E3 --fold 1

# ATR requires GTV masks in $NSCLC_ARTIFACT_DIR/GTV_masks/{pid}_gtv.npy
```

---

## Using Pre-trained Checkpoints

> **Note:** pre-trained checkpoints are not yet published. Until they are,
> reproduce results by retraining from scratch (Steps 3–8). Once uploaded,
> this section will give the exact Hugging Face repository; the snippet below
> shows the intended usage with a placeholder repository id.

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="<your-hf-username>/nsclc-survival-checkpoints",  # placeholder — not yet published
    local_dir="outputs/checkpoints",
    repo_type="model",
)
EOF
```

Then go directly to Step 8 (evaluate.py).

> Checkpoints are expected to be ~2.1 GB total (all experiments, all folds).

---

## Implementation Notes

Correctness-relevant details of the pipeline, for reviewers:

1. **Cox loss sign** (`cox.py`): both losses return the *negative* partial
   log-likelihood normalised by the number of events
   (`-((log_h - lcs) * e).sum() / n_events`), so training minimises the correct
   objective.

2. **Efron tie handling** (`cox.py`): `cox_loss_efron` uses the exact Efron
   per-tie fractional denominator
   `D_j = Σ_{at risk} exp(η) − (j/n_tied)·Σ_{tied} exp(η)`, which corrects the
   Breslow overestimate of the risk set under the many integer-day ties in
   Lung1. Efron is used for E10–E17; Breslow for E1–E9.

3. **No age leakage** (`clinical.py`): `normalise_age_per_fold()` z-scores age
   using *only* training-fold statistics and is applied inside the CV loop.

4. **Wilcoxon fold-count guard** (`metrics.py`): `wilcoxon_compare()` checks
   `len(fci_a) == len(fci_b)` before running, returning `valid=False` with an
   explanation rather than silently comparing COLIPRI (10-fold) against the
   5-fold CNN/clinical experiments.

5. **Integrated Brier Score** (`metrics.py`): uses
   `sksurv.metrics.integrated_brier_score` (Graf 1999) rather than a hand-rolled
   IPCW estimator.

6. **ReLU `inplace=False`** (`cnn.py`, `colipri.py`): all activations avoid
   in-place ops so the stored pre-activations remain available for the GradCAM
   backward pass.

---

## Reproducibility

The pipeline fixes a global seed (`training.seed: 123`), seeds the DataLoader
workers and generators, and saves the CV splits to `outputs/splits/` on the
first run so that train/test assignment is identical on every subsequent run.

With identical data, the same pinned environment, and the same config, re-runs
are closely reproducible. Note that exact OOF C-indices can still differ
slightly between machines because of GPU non-determinism (CuDNN algorithm
selection, non-deterministic atomic reductions), CUDA/cuDNN/PyTorch version
differences, and CPU-vs-GPU execution. Treat small differences in the trailing
decimals as expected rather than as a mismatch. For bit-wise determinism you
would additionally need `torch.use_deterministic_algorithms(True)` and the
corresponding CuDNN flags, at some cost to speed.

---

## Citation

```bibtex
@mastersthesis{shiferaw2026nsclc,
  title  = {Prognostic Deep Learning for Non-Small Cell Lung Cancer:
            Survival Prediction from 3D Thoracic Imaging},
  author = {Shiferaw, Kirubel Biruk},
  school = {[University]},
  year   = {2026},
  note   = {Code: https://github.com/kirubel-Biruk-Shiferaw/NSCLC-Survival_annotation_free-3DCT}
}
```

---

## License

MIT License.  Patient data (TCIA Lung1) is subject to the TCIA Data Usage Agreement.
