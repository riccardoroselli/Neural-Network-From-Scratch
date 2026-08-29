# Neural Network From Scratch in NumPy

A feed-forward neural network library with every backward pass derived and coded by hand on
NumPy, and no automatic differentiation anywhere, applied to four problems: the three MONK
classification benchmarks and ML-CUP25, a 12-input, 4-target regression task with withheld
test labels. Hyperparameters come from a two-phase search — a 97,200-point coarse hold-out
sweep, then a 19,998-point 5-fold cross-validated re-search — costing 14.32 CPU-hours.
MONK-1 and MONK-2 reach 1.000000 ± 0.000000 test accuracy across 9 seeds, MONK-3 0.939558 ±
0.009848; on ML-CUP25 a 10-model ensemble reaches 21.6432 Mean Euclidean Error against
37.4840 for predicting the training mean.

## Data

| Dataset | Train | Test | Inputs | Targets |
|---|---|---|---|---|
| MONK-1 | 124 | 432 | 17 one-of-k | 1 binary |
| MONK-2 | 169 | 432 | 17 one-of-k | 1 binary |
| MONK-3 | 122 | 432 | 17 one-of-k | 1 binary |
| ML-CUP25 | 500 | 1000 blind | 12 continuous | 4 continuous |

MONK's six categorical attributes (cardinalities 3, 3, 2, 3, 4, 2) expand to 17 binary
columns. MONK-3 carries label noise: 6 of its 122 training rows contradict the target rule
`(a5 = 3 and a4 = 1) or (a5 ≠ 4 and a2 ≠ 3)`, while all 432 test rows obey it. The CUP
training file is cut twice from one seed-42 permutation — 425/75 rows for selection,
450/50 for assessment — so the 50-row test set shares no row with selection. CUP inputs and
targets are standardized inside each split, never across it.

## Library

Every component is a `Module` with `forward(X, training)`, `backward(dY)` and
`params_and_grads()`. A `Model` is an ordered list of them: no graph, no tape, sequential
architectures only.

| Module | Contents |
|---|---|
| `nn/layers.py` | `Dense`, gradients `dW = Xᵀ dY`, `db = Σₙ dY`, `dX = dY Wᵀ`, summed not averaged |
| `nn/activations.py` | ReLU, Sigmoid (input clipped to ±20), Tanh, Identity, Softmax |
| `nn/losses.py` | MSE, MEE (‖·‖₂ clipped at 1e-12), BinaryCrossEntropy, CrossEntropy |
| `nn/optim.py` | SGD, SGD with momentum, Adam — state keyed by network position, not `id()` |
| `nn/regularizers.py` | L1 and L2 on weights only, applied as ordinary gradients before the step |
| `nn/dropout.py` | Inverted dropout with a private RNG restarted by `reset()` |
| `nn/callbacks.py` | Six hooks; `EarlyStopping` snapshots and restores the best state |
| `nn/model.py` | Sequential container, state save/restore, reset |

Both cross-entropies are fused with their activation and are correct only as a pair: BCE
returns `(1/N)(p − y)/(p(1 − p))` against a sigmoid backward of `p(1 − p)`, and categorical
CE returns `(p − y)/N` with respect to the logits, leaving `Softmax.backward` a pass-through.
Derivations are in [`report/presentation.pdf`](report/presentation.pdf).

## Model selection

The coarse phase sweeps the full grid once on an 80/20 hold-out split — cheap, biased, and
only meant to locate a region. `refine_grid.py` then spans the min and max of each continuous
hyperparameter across the top-ranked coarse configurations and spaces points linearly across
it; the fine phase re-searches that grid under 5-fold cross-validation, stratified for MONK
and plain K-fold for CUP. Architecture and activation are never narrowed, and the winner
often moves: MONK-1's coarse best was two ReLU layers of 4 units, its fine best a single
sigmoid layer of 8. No test set is visible to either phase.

| Search | Coarse | Fine | CPU time |
|---|---|---|---|
| MONK-1 | 31,104 | 1,728 | 1.32 h |
| MONK-2 | 31,104 | 2,304 | 1.64 h |
| MONK-3 | 31,104 | 9,216 | 1.41 h |
| ML-CUP25 | 3,888 | 6,750 | 9.94 h |

MONK's coarse grid is 6 architectures × 3 activations × 3 dropout rates × 4 L2 values × 2
optimizers × 6 learning rates × 3 momenta × 2 batch sizes × 2 epoch budgets; CUP's drops the
epoch axis and fixes SGD. Fine grids come from the top 10 coarse configurations, top 20 for
MONK-3, all on seed 42. The logs hold 118,868 rows against 117,198 grid points, the excess
being earlier partial searches left in the MONK-1 and CUP coarse files.

| Selected model | Architecture | Params | Activation | Optimizer | L2 | Batch | Epochs |
|---|---|---|---|---|---|---|---|
| MONK-1 | 17-8-1 | 153 | sigmoid | Adam, lr 0.2 | 0 | 16 | 1000 |
| MONK-2 | 17-12-1 | 229 | ReLU | Adam, lr 0.04 | 0 | 16 | 1000 |
| MONK-3 | 17-12-8-4-1 | 361 | ReLU | Adam, lr 0.04 | 3e-7 | 32 | 800 |
| ML-CUP25 | 12-64-32-16-4 | 3508 | tanh | SGD, lr 0.01, momentum 0.8 | 1e-4 | 360 | 1000 |

Dropout is 0 in all four, having collapsed to a single value in every fine grid. The three
MONK rows are the top of their `fine_summary.csv`; the CUP row was set by hand afterwards
over the search's own winner (lr 0.06, L2 8e-4, batch 64, 800 epochs), and it produced every
CUP number below.

## Results

Assessment retrains from scratch on data untouched by selection. MONK takes its epoch count
from 5-fold CV on the training set alone, then retrains on the full training set for each of
9 seeds and scores the 432-row official test set with early stopping switched off, so test
performance cannot pick an epoch.

| Problem | Test accuracy (9 seeds) | Ensemble | Test loss (BCE) |
|---|---|---|---|
| MONK-1 | 1.000000 ± 0.000000 | 1.000000 | 0.000017 ± 0.000024 |
| MONK-2 | 1.000000 ± 0.000000 | 1.000000 | 0.000002 ± 0.000005 |
| MONK-3 | 0.939558 ± 0.009848 | 0.949074 | 0.515718 ± 0.127881 |

MONK-1 and MONK-2 are solved exactly — every seed classifies all 432 test rows correctly.
MONK-3 trains to 0.999089 ± 0.002576 accuracy against 0.939558 on test: the gap is its 4.92
percent label noise being memorized, and cross-validation puts the honest stopping point at
epoch 57 rather than the 800 trained.

CUP is scored by 5-fold CV on the 450-row internal split, which fixes the epoch budget at
720, then by a single retrain measured on the untouched 50 rows. In original target units:

| Quantity | MEE |
|---|---|
| Validation folds | 22.6445 ± 2.3910 |
| Internal test, single model | 22.5881 |
| Internal test, ensemble of 10 | 21.6432 |
| Predicting the training mean | 37.4840 |

Validation and internal test agree to within 0.06 MEE, the main evidence that selection did
not overfit its own validation data; ensembling is worth about one further point. The
submission averages 10 models (seeds 42–51) retrained on all 500 rows for 720 epochs.
Learning curves are drawn inline by the assessment notebooks and not stored as files.

## Project structure

```
nn/                             The network, NumPy only
  core.py                       Module base class: forward, backward, params_and_grads
  layers.py                     Dense, with hand-derived dW, db and dX
  activations.py                ReLU, Sigmoid, Tanh, Identity, Softmax
  losses.py                     MSE, MEE, BinaryCrossEntropy, CrossEntropy
  metrics.py                    Accuracy, Precision, Recall, F1Score, MEE, MSE
  optim.py                      SGD, SGDMomentum, Adam
  regularizers.py               L1 and L2, weights only
  dropout.py                    Inverted dropout with a private RNG stream
  initializers.py               xavier_uniform, he_uniform, zeros
  callbacks.py                  Callback protocol and EarlyStopping
  model.py                      Sequential container: forward, step, state, reset

training/                       Fitting and hyperparameter search
  trainer.py                    fit and evaluate: mini-batch loop, callback dispatch
  dataloader.py                 BatchIterator: per-epoch shuffling and batching
  history.py                    History: per-epoch metric log
  config.py                     load_config, expand_grid, get/set_by_path
  model_factory.py              build_model_from_cfg: config dictionary to Model
  holdout_cv.py                 One train/validation split, normalized within it
  kfold_cv.py                   K-fold CV, normalized per fold, task auto-detection
  gridsearch.py                 Process-pool execution, aggregation, ranking
  refine_grid.py                Coarse top-k to fine grid ranges
  model_selection.py            run_two_phase_selection: coarse, refine, fine

evaluation/
  ensemble_utils.py             Majority vote, best-epoch from CV, curve plotting

data/
  data_handler/data_loader.py   load_monk with 1-of-k encoding, load_cup, normalize
  MONK/MONK1/monks-1.train      124 rows
  MONK/MONK1/monks-1.test       432 rows
  MONK/MONK2/monks-2.train      169 rows
  MONK/MONK2/monks-2.test       432 rows
  MONK/MONK3/monks-3.train      122 rows, 6 of them mislabelled
  MONK/MONK3/monks-3.test       432 rows
  CUP/ML-CUP25-TR.csv           500 labelled rows: 12 inputs, 4 targets
  CUP/ML-CUP25-TS.csv           1000 blind rows
  CUP/internal_train_80.csv     425 rows, the selection training set
  CUP/internal_test_20.csv      75 rows, withheld during selection
  CUP/internal_train_90.csv     450 rows, the assessment training set
  CUP/internal_test_10.csv      50 rows, the final internal test set

experiments/                    Configs and recorded results, 176 MiB of CSVs
  MONK/common.py                Data loading, model building, selection entry point
  MONK/configs/monk1.yaml       Base config plus the nine-axis coarse grid
  MONK/configs/monk2.yaml       As above, MONK-2
  MONK/configs/monk3.yaml       As above, MONK-3
  MONK/results/monk1/           The eight files every search writes:
    coarse_runs.csv               one row per configuration and seed (31,473)
    coarse_summary.csv            aggregated and ranked (31,104)
    fine_grid.json                value ranges derived from the coarse top-k
    refine_report.json            per-axis min, max, steps and clip bounds
    fine_config.json              base config plus the fine grid
    fine_runs.csv                 one row per fine configuration (1,728)
    fine_summary.csv              aggregated and ranked (1,728)
    best_config.json              the winner, loaded by the assessment notebooks
  MONK/results/monk2/           Same eight files: 31,104 coarse, 2,304 fine
  MONK/results/monk3/           Same eight files: 31,104 coarse, 9,216 fine
  CUP/common.py                 As MONK, plus create_internal_split
  CUP/configs/cup_coarse.yaml   Base config plus the eight-axis coarse grid
  CUP/results/cup_exp_1/        Same eight files: 5,189 coarse rows, 6,750 fine
  CUP/submission/iCavalli_ML-CUP25-TS.csv
                                1000 ensemble predictions, four-line ML-CUP header

notebooks/
  run_monk_experiments.ipynb    Both search phases, three MONK problems
  run_cup_experiments.ipynb     Both search phases, ML-CUP25
  run_monk_test.ipynb           9-seed retraining, curves, majority-vote ensembles
  run_cup_test.ipynb            CUP cross-validation, internal test, ensemble, submission

tests/                          148 tests, about 5 seconds
  gradcheck.py                  numeric_gradient, relative_error, check_* helpers
  test_gradients.py             31 finite-difference checks of every backward pass
  test_optimizers.py            15 update-rule checks against reference formulas
  test_callbacks.py             12 EarlyStopping tracking and restoration checks
  test_components.py            45 checks of metrics, initializers, config, normalization
  test_gridsearch.py            21 checks of ranking, aggregation and grid refinement
  test_pipeline.py              24 end-to-end, CV, determinism and data-loader checks

report/presentation.pdf         20 slides: formulation, derivations, results
requirements.txt                The five pinned dependencies
```

Each `experiments/*/results/` directory holds the runs and ranked summaries of both phases,
the derived fine grid with its report, and `best_config.json` — the file every assessment
notebook loads.

## Running it

Python 3.13, recorded on 3.13.7. Install, then check the build with no data beyond the repo:

```bash
pip install -r requirements.txt          # numpy, scikit-learn, matplotlib, PyYAML, jupyterlab
python -m unittest discover -s tests -t . # 148 tests, about 5 s
python -m tests.test_gradients            # 31 checks vs central differences, tolerance 1e-6
```

The gradient report prints a max relative error per component: 21 of the 23 pass at 6.5e-8 or
better, and two read `expected mismatch` by design — the clipped sigmoid's saturated region,
and `CrossEntropy` measured unfused. scikit-learn is used for `train_test_split` and the fold
splitters and nothing else; no estimator, metric or transformer of its own enters the models.

```bash
jupyter lab notebooks/
```

| Notebook | Produces | Cost |
|---|---|---|
| `run_monk_experiments.ipynb` | Both search phases, three MONK problems | 4.4 CPU-hours |
| `run_cup_experiments.ipynb` | Both search phases, ML-CUP25 | 9.9 CPU-hours |
| `run_monk_test.ipynb` | 9-seed retraining, curves, majority-vote ensembles | 79 s |
| `run_cup_test.ipynb` | CUP cross-validation, internal test, ensemble, submission | 18 s |

Each notebook locates the project root on open, so Jupyter may start anywhere. Search costs
are summed CPU time over `n_jobs=-1` workers; divide by cores for wall-clock. Both phases
delete their run CSV first, so point `out_dir` elsewhere to keep an old search, and run
`run_cup_test.ipynb` from the top — its submission cells need the epoch count computed above
them.

## Notes

Initialization, mini-batch shuffling, dropout masks and CV splits are all seeded: four tests
in `tests/test_pipeline.py` assert bit-identical reruns and a fifth asserts that an unseeded
fit differs. Two consequences are deliberate: every
fold of a k-fold run starts from identical weights, so fold spread excludes initialization
sensitivity, and all layers of a model share one seed value.

Re-running assessment today reproduces MONK-1, MONK-2 accuracy and both CUP test figures
exactly; MONK-3 lands a point lower (0.930298 ± 0.014246), a code change between revisions
rather than nondeterminism. The search CSVs and the committed submission came from earlier
revisions and will not reproduce byte for byte.

## Authors

Team *iCavalli*, Machine Learning, University of Pisa.

- Lorenzo Albani, l.albani2@studenti.unipi.it
- Tommaso Maitino, t.maitino@studenti.unipi.it
- Riccardo Roselli, r.roselli1@studenti.unipi.it

