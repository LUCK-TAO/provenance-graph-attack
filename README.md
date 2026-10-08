# Provenance Graph Attack

A detector-agnostic black-box adversarial attack framework for provenance-graph intrusion detection systems. The repository integrates multiple provenance detectors behind a common query interface and evaluates semantic, directed graph-edit attacks under explicit query and edit budgets.

## Included detectors

- MAGIC
- FLASH
- Orthrus
- STGAN
- ProvGuard

The unified adapters are under `Detector/`. Detector-specific source code and setup notes remain in their corresponding directories.

## Main capabilities

- semantic action masks for valid directed provenance edits;
- local/global candidate-node construction;
- macro-actions with one detector query per edit batch;
- PPO, LinUCB, Double DQN, greedy, random, and benign-mimicry policies;
- detector-native thresholds, scores, labels, and evaluation logic;
- initial-TP targeting for cross-detector experiments;
- ASR, AAB, AQT, TP-reduction, amplification, and graph-perturbation metrics;
- reproducible multi-seed PowerShell experiment scripts.

## Repository layout

```text
ATTACK.py                  Unified experiment entry point
Detector/                  Detector adapters and native-query workers
MAGIC_main/                Shared attack framework and MAGIC integration
Flash_IDS_main/            FLASH implementation
Orthrus_official/          Orthrus source and attack integration
STGAN_master/              STGAN implementation
ProvGuard_main/            ProvGuard implementation
tests/                     Lightweight regression tests
run_*.ps1                  Reproducible experiment launchers
EXPERIMENT_REPRODUCTION.md Experiment conventions and commands
```

## Data and model files

Datasets, database dumps, trained weights, embeddings, detector artifacts, attack states, and raw experiment outputs are intentionally **not included**. They are large and may be subject to separate distribution terms.

Follow the README in each detector directory to obtain the corresponding public datasets and pretrained artifacts. Place downloaded material in the paths expected by that detector. The root `.gitignore` prevents these local files from being committed accidentally.

## Basic usage

Create detector-specific Python/Conda environments according to the included detector READMEs, then inspect the unified command line:

```bash
python ATTACK.py --help
```

The `run_*.ps1` files provide the exact experiment arguments used for the supported dataset/detector combinations. See `EXPERIMENT_REPRODUCTION.md` for additional details.

## Reproducibility notes

- Detector evaluation is delegated to each detector's native scoring path.
- Attack-time target selection and rewards use only observable detector output unless an experiment explicitly targets the fixed initial native TP pool.
- Random seeds, thresholds, query counts, edit counts, and final metrics are stored in each generated experiment summary.

## Third-party code and licenses

This repository combines research code from several detector projects with a unified attack framework. Copyright and license terms remain with the respective authors. Existing LICENSE and README files are retained inside the detector directories. No single repository-wide license is asserted over third-party components.

Use this code only for authorized security research and evaluation.

