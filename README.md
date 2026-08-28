# Neural network from scratch in NumPy

A feed-forward neural network library implemented directly on NumPy, with every backward pass
derived and coded by hand and no automatic differentiation. It is used here to run a two-phase
hyperparameter search on four datasets: the three MONK classification benchmarks, and CUP, a
four-target regression task whose test labels are withheld.

## 1. At a glance

`nn/` provides dense layers, five activations, four losses, three optimizers, L1 and L2
regularization, inverted dropout, two initializers and a callback system. The hand-derived
gradients are checked against central finite differences: 21 checks agree to better than
1e-7, worst case 6.5e-8, plus two deliberate deviations described in section 7.

Hyperparameters were chosen in two phases: a coarse hold-out sweep over a large grid, then a
5-fold cross-validated re-search over a narrowed grid. Across the four problems, 117,198
training runs were recorded, totalling 14.32 hours of CPU time.

MONK-1 and MONK-2 reach 1.000000 +/- 0.000000 test accuracy over 9 seeds; MONK-3, the noisy
variant, reaches 0.939558 +/- 0.009848. On CUP the Mean Euclidean Error in original
target units is 22.6445 +/- 2.3910 in cross-validation, 22.5881 on the internal test set and
21.6432 for an ensemble of 10, against 37.4840 for predicting the training mean. Section 8
gives the full tables.

## 2. What "from scratch" means here

Everything constituting the model and its training is implemented here with NumPy only: the
forward pass, the analytical backward pass, the parameter updates, the losses and metrics,
and the standardization of inputs and targets. Three other libraries support the experiments
and contribute no part of the model.

| Library | Purpose | Location |
|---|---|---|
| NumPy | All numerical computation, including the network | Throughout |
| scikit-learn | Dataset splitting only: `train_test_split`, `KFold`, `StratifiedKFold` | `training/holdout_cv.py`, `training/kfold_cv.py`, `notebooks/run_cup_test.ipynb` |
| matplotlib | Learning-curve plots | `evaluation/ensemble_utils.py`, notebooks |
| PyYAML | Reading the configuration files | `training/config.py` |

scikit-learn decides which rows go into which fold and does nothing else. No estimator,
metric, preprocessing transformer or scoring function from it is used. Standardization is in
`data/data_handler/data_loader.py`; all losses and metrics are in `nn/`.

## 3. The framework

**Modules.** Every component in the forward and backward pass is a `Module` (`nn/core.py`)
with three methods: `forward(X, training=True)`, the flag existing only so dropout can differ
at inference; `backward(dY)`, taking the gradient with respect to this module's output and
returning it with respect to its input; and `params_and_grads()`, empty by default. Each
caches what its backward needs: `Dense` its input, `ReLU` its mask, `Sigmoid` and `Tanh` their
outputs. A `Model` is an ordered list of modules with no graph and no tape, so position in the
list defines the topology and only sequential architectures are possible.

**Dense** computes `Y = X W + b`. `dW` and `db` are sums over the batch, not means: the `1/N`
is applied once in the loss backward, so batch-size normalization is never duplicated.

```
dW = X^T dY          db = sum_n dY          dX = dY W^T
```

**MSE**, used for CUP regression. The per-sample term sums squared error across output
dimensions, then averages over samples.

```
L = (1/N) sum_n ||y_hat_n - y_n||^2          dL/dy_hat = (2/N)(y_hat - y)
```

**MEE**, used to score the regression task. The gradient is a unit vector from target to prediction
scaled by `1/N`. Residual magnitude does not scale it, so unlike MSE one badly predicted
sample cannot dominate an update. There is no derivative at a zero residual, so the norm is
clipped at 1e-12.

```
L = (1/N) sum_n ||y_hat_n - y_n||_2
dL/dy_hat_n = (1/N) (y_hat_n - y_n) / ||y_hat_n - y_n||_2
```

**Cross-entropy** is fused with its activation in both forms. Binary cross-entropy returns
`(1/N)(p-y)/(p(1-p))` and the sigmoid backward multiplies by `p(1-p)`, reducing the product to
`(p-y)/N` exactly; the sigmoid clips its input to `[-20, 20]`, keeping `p` inside
`[2.06e-9, 1 - 2.06e-9]` and clear of the loss's own 1e-12 clip. Categorical cross-entropy
returns `(p-y)/N`, the gradient with respect to the logits rather than `p`, so
`Softmax.backward` is a pass-through. Each pair is correct only as a pair.

**Optimizers** hold no reference to the model; `step(modules)` pulls `(parameter, gradient)`
pairs and mutates the arrays in place.

```
SGD                theta <- theta - lr * g
SGD + momentum     v <- mu * v - lr * g ;  theta <- theta + v
Adam               m <- b1 m + (1-b1) g ;  v <- b2 v + (1-b2) g^2
                   m_hat = m / (1 - b1^t) ;  v_hat = v / (1 - b2^t)
                   theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
```

Momentum and Adam key their buffers on the parameter's position in the network, the pair
(module index, parameter index), rather than on `id(param)`: object identity is unique only
among live objects, so reinitializing a layer could hand a new array the address of a freed
one and with it another parameter's momentum.

**Regularizers** expose `penalty(modules)`, added to the loss in `Model.compute_loss`, and
`add_gradients(modules)`, applied in `Model.step` before the optimizer so the term becomes an
ordinary gradient. L2 is `0.5 * lambda * sum ||W||^2` with `dW += lambda * W`; L1 is
`lambda * sum |W|` with `dW += lambda * sign(W)`. Both act on weights only, never biases.

**Callbacks** define six hooks (train, epoch and batch begin and end); a truthy return requests
a stop, combined with a logical OR. `EarlyStopping` snapshots the best state through
`Model.get_state()` and restores it when training ends, whether patience ran out or the epoch
limit was reached.

**Extending.** Subclass the relevant base, implement its contract, and register it in
`training/model_factory.py` to reach it from a config file. New optimizers should use
`_iter_keyed_params`, and new gradients a check in `tests/test_gradients.py`.

## 4. Key numbers

| Dataset | Train | Test | Inputs | Targets |
|---|---|---|---|---|
| MONK-1 | 124 | 432 | 17 after encoding | 1 binary |
| MONK-2 | 169 | 432 | 17 after encoding | 1 binary |
| MONK-3 | 122 | 432 | 17 after encoding | 1 binary |
| CUP training | 500 | | 12 continuous | 4 continuous |
| CUP blind test | | 1000 | 12 continuous | withheld |

MONK inputs are six categorical attributes with cardinalities 3, 3, 2, 3, 4 and 2, encoded
one-of-k into 17 binary columns. The CUP training set is split internally twice: 80/20 (425
and 75 rows) for selection, 90/10 (450 and 50 rows) for assessment. Both come from the same
seed-42 permutation, so the 50-row assessment set contains no row used during selection.

| Problem | Architecture | Parameters | Activation | Optimizer | L2 | Batch | Epochs |
|---|---|---|---|---|---|---|---|
| MONK-1 | 17-8-1 | 153 | sigmoid | Adam, lr 0.2 | 0 | 16 | 1000 |
| MONK-2 | 17-12-1 | 229 | ReLU | Adam, lr 0.04 | 0 | 16 | 1000 |
| MONK-3 | 17-12-8-4-1 | 361 | ReLU | Adam, lr 0.04 | 3e-7 | 32 | 800 |
| CUP | 12-64-32-16-4 | 3508 | tanh | SGD, lr 0.01, momentum 0.8 | 1e-4 | 360 | 1000 |

Parameter counts include biases. Dropout is 0 in all four. Early stopping patience is 40
epochs for MONK, 50 for CUP.

| Experiment | Coarse | Fine | CPU time |
|---|---|---|---|
| MONK-1 | 31,104 | 1,728 | 1.32 h |
| MONK-2 | 31,104 | 2,304 | 1.64 h |
| MONK-3 | 31,104 | 9,216 | 1.41 h |
| CUP | 3,888 | 6,750 | 9.94 h |
| Total | 97,200 | 19,998 | 14.32 h |

The MONK coarse grid is 6 architectures x 3 activations x 3 dropout rates x 4 L2 values x 2
optimizers x 6 learning rates x 3 momentum values x 2 batch sizes x 2 epoch budgets = 31,104.
The CUP grid drops the epoch axis and fixes SGD, giving 3,888. Fine grids differ because each
is derived from its own coarse results, using the top 10 configurations for MONK-1, MONK-2 and
CUP, and the top 20 for MONK-3. CPU time is the sum of `train_time_sec` over all runs;
wall-clock is roughly that divided by the worker count, and the recorded runs used 8
processes. Every search used a single seed, 42. Assessment used 9 seeds for MONK (9, 11, 35,
42, 51, 68, 74, 81, 99) and 10 for the CUP ensembles, 42 through 51.

## 5. Code structure

```
nn/                  The network, NumPy only
  core.py            Module base class: forward, backward, params_and_grads
  layers.py          Dense, with hand-derived dW, db and dX
  activations.py     ReLU, Sigmoid, Tanh, Identity, Softmax
  losses.py          MSE, MEE, BinaryCrossEntropy, CrossEntropy
  metrics.py         Accuracy, Precision, Recall, F1Score, MEE, MSE
  optim.py           SGD, SGD with momentum, Adam
  regularizers.py    L1 and L2, weights only
  dropout.py         Inverted dropout with its own RNG stream
  initializers.py    Xavier uniform, He uniform, zeros
  callbacks.py       Callback protocol and EarlyStopping
  model.py           Sequential container, state save and restore, reset
training/            Fitting and hyperparameter search
  trainer.py         Mini-batch loop, evaluation, callback dispatch
  dataloader.py      Shuffling and mini-batching
  history.py         Per-epoch metric log
  config.py          YAML/JSON loading, Cartesian grid expansion
  model_factory.py   Builds a Model from a config dictionary
  holdout_cv.py      Train/validation split, normalized within the split
  kfold_cv.py        K-fold CV, normalized per fold, with task detection
  gridsearch.py      Parallel grid execution and result aggregation
  refine_grid.py     Derives the fine grid from the coarse results
  model_selection.py Two-phase coarse-to-fine driver
evaluation/          Majority voting, best-epoch estimation, curve plotting
experiments/         Per-problem configs, recorded results, CUP submission
notebooks/           The four experiment drivers
data/                Datasets, plus data_handler/ with the parsing code
tests/               148 tests, about 5 seconds
```

`experiments/` holds 184 MB, almost all recorded search CSVs; every other directory is under
100 KB. Each results directory holds eight files: `coarse_runs.csv` and `fine_runs.csv` (one
row per configuration and seed), `coarse_summary.csv` and `fine_summary.csv` (aggregated and
ranked), `fine_grid.json` and `refine_report.json` (how the fine grid was derived),
`fine_config.json`, and `best_config.json`.

## 6. Experimental methodology

**Preprocessing.** MONK inputs are encoded one-of-k and otherwise untouched, being already
binary. CUP inputs and targets are standardized to zero mean and unit variance: standardizing
the inputs improves conditioning, and standardizing the targets matters because their
per-dimension standard deviations are 14.5, 14.6, 22.8 and 22.8, so the two wider ones would
otherwise dominate the loss. Statistics are computed on the training portion of the current
split and applied to the held-out portion, inside the cross-validation functions rather than
in the caller, so a validation fold cannot contribute to the statistics that scale it.

**Why two phases.** Searching the full space at fine resolution is not affordable: MONK has
nine grid axes and CUP eight, and evaluating a candidate properly needs cross-validation. The
coarse phase sweeps the whole grid with one 80/20 hold-out split, a weak but cheap estimate
whose only job is to locate a promising region. The fine phase re-searches with 5-fold
cross-validation over a grid built by `refine_grid.py`: for each continuous hyperparameter it
takes the smallest and largest value among the best coarse configurations, places linearly
spaced points across that range, rounds to one significant digit and clips to sane bounds.
Architecture and activation are not narrowed, so all six architectures and three activations
are re-explored; MONK-1's coarse winner was a two-layer ReLU network with 4 units per layer,
its fine winner a single sigmoid layer of 8. Dropout was 0 in the leading configurations of
all four problems, collapsing that axis to one value in every fine grid.

**Validation.** Classification uses stratified splits so class balance is preserved per fold;
regression uses plain K-fold. `kfold_cv.py` chooses automatically, treating multi-output or
floating-point targets as regression. No test set is visible to selection: the MONK searches
read only `monks-N.train`, the CUP search only the 425-row internal split.

**Assessment.** Selection scores are the maximum over thousands of candidates and biased
upward, so they say nothing reliable about generalization. Assessment retrains from scratch on
data untouched until that moment. For MONK the epoch count comes from 5-fold cross-validation
on the training set alone, per seed then averaged; the model is retrained on the full training
set for each of 9 seeds and evaluated on the 432-row official test set. The test set is passed
to the trainer only so its curves can be logged, and early stopping is disabled so it cannot
pick an epoch using test performance. For CUP, 5-fold cross-validation on the 450-row internal
split gives the validation MEE and an average best epoch of 720; one model is retrained on all
450 rows for that many epochs and scored once on the 50-row test set.

**Ensembling and the submission.** MONK combines its 9 models by majority vote, mean
probability breaking ties. CUP averages 10 models in standardized space before mapping back,
worth about one MEE point. The submission comes from 10 models retrained on the whole 500-row
training set for 720 epochs, with no held-out portion since selection and assessment are
finished; predictions on the 1000 blind rows are averaged, denormalized and written with the
required four-line header.

## 7. Running the experiments

Python 3.13; the recorded runs used 3.13.7. From the repository root:

```bash
pip install -r requirements.txt
```

This installs NumPy 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.6, PyYAML 6.0.3 and JupyterLab
4.4.9, the versions the experiments were run against. To check the installation:

```bash
python -m unittest discover -s tests -t .
```

Expect 148 tests and `OK` in about 5 seconds, with no data needed beyond the repository.
`python -m tests.test_gradients` runs the gradient checks alone, printing the maximum relative
error of every backward pass: 31 tests, no failures. Two rows read `expected mismatch`, the
saturated region of the clipped sigmoid where the code returns about 2e-9 rather than 0, and
`CrossEntropy` measured alone rather than fused with `Softmax`.

The experiments run from notebooks, each opening with a bootstrap cell that locates the project
root and changes into it, so they work whether Jupyter starts in `notebooks/` or at the root.

```bash
jupyter lab notebooks/
```

| Notebook | Purpose | Cost |
|---|---|---|
| `run_monk_experiments.ipynb` | Model selection, all three MONK problems | 4.4 CPU-hours |
| `run_cup_experiments.ipynb` | Model selection, CUP | 9.9 CPU-hours |
| `run_monk_test.ipynb` | Retrains 9 seeds per problem, evaluates, plots, ensembles | 79 s |
| `run_cup_test.ipynb` | CUP cross-validation, internal test, ensemble, submission | 18 s |

Selection costs are summed CPU time, so divide by your core count for wall-clock; with the 8
processes used originally that is roughly ten minutes per MONK search and an hour for the CUP
fine phase. Assessment timings were measured end to end on an Apple M4 with 10 cores. Both
search phases clear their run CSV before starting, so re-running a search overwrites the
results in that `out_dir` rather than appending; point `out_dir` somewhere new to keep them.
The submission comes from the last four cells of `run_cup_test.ipynb`, which depend on
`avg_best_epoch` computed earlier, so run that notebook from the top.

## 8. Results

Mean and standard deviation over the 9 seeds. Loss is binary cross-entropy; MSE is included as
the measure usually quoted for these benchmarks.

| Problem | Train accuracy | Test accuracy | Train loss | Test loss | Train MSE | Test MSE |
|---|---|---|---|---|---|---|
| MONK-1 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.000005 +/- 0.000004 | 0.000017 +/- 0.000024 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| MONK-2 | 1.000000 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.000000 +/- 0.000001 | 0.000002 +/- 0.000005 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| MONK-3 | 0.999089 +/- 0.002576 | 0.939558 +/- 0.009848 | 0.002325 +/- 0.006442 | 0.515718 +/- 0.127881 | 0.000729 +/- 0.002061 | 0.055658 +/- 0.010292 |

| Majority-vote ensemble | Train accuracy | Test accuracy | Test loss |
|---|---|---|---|
| MONK-1 | 1.000000 | 1.000000 | 0.000017 |
| MONK-2 | 1.000000 | 1.000000 | 0.000002 |
| MONK-3 | 1.000000 | 0.949074 | 0.227400 |

MONK-1 and MONK-2 are solved exactly: every seed classifies all 432 test rows correctly.
MONK-3 is the noisy problem. Against the target rule `(a5 = 3 and a4 = 1) or (a5 != 4 and
a2 != 3)`, 4.92 percent of training rows are mislabelled while the test set is clean, and the
distance between 99.9 percent training accuracy and 93.96 percent test accuracy is that noise
being learned. Cross-validation put the stopping point at epoch 57, far short of the 800 the
assessment trains.

CUP, five-fold cross-validation on the 450-row internal split, standardized units:

| | Loss (MSE) | MEE |
|---|---|---|
| Training folds | 1.5509 +/- 0.6031 | 1.1113 +/- 0.2172 |
| Validation folds | 1.9830 +/- 0.4454 | 1.2679 +/- 0.1556 |

Best epochs per fold were 1000, 13, 1000, 600 and 991, averaging to 720, which sets the epoch
budget for all later retraining. In original target units:

| Quantity | MEE |
|---|---|
| Training folds | 19.9225 +/- 3.0438 |
| Validation folds | 22.6445 +/- 2.3910 |
| Internal test, single model | 22.5881 |
| Internal test, ensemble of 10 | 21.6432 |
| Predicting the training mean | 37.4840 |

Validation MEE of 22.6445 and internal test MEE of 22.5881 agree closely, the main evidence
that selection did not overfit its own validation data.

The three MONK configurations in section 4 are the top-ranked rows of their respective
`fine_summary.csv`. The CUP configuration is not: the fine search ranked first a configuration
with learning rate 0.06, L2 8e-4, batch size 64 and 800 epochs, and the one in the table, which
produced every CUP number here and the submitted predictions, was set by hand afterwards. Both
share the same 12-64-32-16-4 tanh architecture and momentum. Learning curves are drawn inline
by the two assessment notebooks and are not stored as image files.

## 9. Reproducibility

Randomness enters at four points, all seeded: weight initialization
(`np.random.default_rng(seed)` per layer), mini-batch shuffling (a fresh generator each epoch,
seeded from the fit seed plus the epoch index), dropout masks (a per-module generator restarted
by `reset()`), and cross-validation splits (an explicit `random_state`). Running the same
configuration twice on the same machine gives identical numbers; four tests in
`tests/test_pipeline.py` assert this, and a fifth asserts the opposite for a fit called without
a seed.

Two properties are deliberate and matter when reading the standard deviations: every fold of a
k-fold run starts from the same initial weights, since `reset()` reinitializes from the same
fixed seed, so fold-to-fold spread excludes sensitivity to initialization; and all layers
within a model are constructed with the same seed value.

Re-executing both assessment notebooks against the current code gives:

MONK-1 test accuracy and loss, MONK-2 test accuracy, CUP internal test MEE (22.5881) and CUP
ensemble MEE (21.6432) reproduce exactly. The rest differ as follows.

| Quantity | Recorded | Current run |
|---|---|---|
| MONK-2 test loss | 0.000002 +/- 0.000005 | 0.000001 +/- 0.000001 |
| MONK-3 test accuracy | 0.939558 +/- 0.009848 | 0.930298 +/- 0.014246 |
| MONK-3 ensemble test accuracy | 0.949074 | 0.939815 |
| CUP validation MEE | 22.6445 +/- 2.3910 | 22.6426 +/- 2.3918 |

MONK-2 differs in the sixth decimal of a loss that is zero to within rounding, with identical
accuracy, and the CUP fold aggregate in the third decimal. MONK-3 is the one real discrepancy,
about a point of test accuracy, caused by a code change between the recorded run and the
current revision rather than by nondeterminism: the earlier code also gives the current figure
when run today, and both show the same overfitting at the same order of magnitude.

Three caveats. The recorded search CSVs were produced by an earlier revision and a fresh search
will not reproduce them byte for byte. The committed submission predates the seeding of
mini-batch shuffling in that code path, so it cannot be regenerated exactly, though a run today
is deterministic. The cross-validated stopping epoch for MONK-1 is now 935 rather than the
recorded 800, because early stopping is disabled during assessment so the test set cannot
influence the chosen epoch, which lengthens the fold histories the figure comes from; that
value only positions a marker on the plots.

## 10. Authors

Team iCavalli.

- Lorenzo Albani, l.albani2@studenti.unipi.it
- Tommaso Maitino, t.maitino@studenti.unipi.it
- Riccardo Roselli, r.roselli1@studenti.unipi.it
