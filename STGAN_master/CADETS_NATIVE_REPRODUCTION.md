# STGAN CADETS native reproduction

This reproduction is independent of the repository-level MAGIC and FLASH
adapters.  It uses the released STGAN CADETS split, Word2Vec vectors, 50 GAT
checkpoints, 50 TGN checkpoints, 50 self-attention checkpoints, and the
released two-hop evaluation helper.

## Environment

- Python 3.9
- PyTorch 1.12.0+cu113
- PyTorch Geometric 2.1.0
- Gensim 4.3.0
- NumPy 1.24.4
- SciPy 1.10.1

SciPy is pinned because Gensim 4.3.0 imports `scipy.linalg.triu`, which was
removed in SciPy 1.13.  `model/__init__.py` exports the released model classes
so that the original `from model import ...` statements resolve correctly.

## Run

From `STGAN_master`:

```powershell
D:\App\anaconda\envs\STGAN\python.exe reproduce_cadets_native.py prepare
D:\App\anaconda\envs\STGAN\python.exe reproduce_cadets_native.py evaluate --device cuda:0
```

The prepare step replaces the released evaluator's multi-gigabyte in-memory
Pandas merge with an equivalent SQLite merge/deduplication cache.  It preserves
timestamp ordering, full-row `drop_duplicates` semantics, node-type labels,
positional Word2Vec aggregation, event directions, and event times.

## Outputs

- `data/cadets/cadets_stgan_official_rows.sqlite3`: on-disk deduplication cache.
- `data/cadets/cadets_stgan_official_test.npz`: standalone detector input.
- `eval_result/stgan_cadets_official_reproduction.json`: metric report.
- `eval_result/stgan_cadets_official_alert_uuids.json`: 353 persistent alerts.
- `eval_result/stgan_cadets_official_initial_tp_uuids.json`: 11 direct TP nodes.

## Reproduction status

The released-code pipeline is reproducible and the original helper was
cross-checked exactly.  The resulting released two-hop metrics are:

| Metric | Reproduced | Paper |
|---|---:|---:|
| Precision | 0.999378 | 0.9840 |
| Accuracy | 0.999961 | 0.9994 |
| Recall | 0.999533 | 0.9983 |
| F1 | 0.999456 | 0.9911 |

The supplied artifacts therefore reproduce the released implementation but do
not exactly reproduce the paper table.  The local source archive has no Git
history and its README contains a placeholder clone URL, so the precise paper
commit and the reason for the remaining metric difference cannot be verified
from the supplied release alone.

