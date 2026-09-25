# TwinDyn: label-free microenvironment digital twin for ccRCC survival prognosis

Code release for the manuscript *Self-Supervised Digital Twin Modeling of Tumor
Microenvironment Dynamics for Survival Prognosis in Kidney Cancer*.

## Overview

The manuscript argues that the transfer tax of microenvironment prognostic models
is structural: a model that compresses each tumour into one static composition
vector and fits that vector to a single cohort's outcome absorbs the cohort's
acquisition properties as if they were biology. It proposes **TwinDyn**, which
treats the microenvironment as a latent state that evolves along an ordered axis
of compartments and learns the transition law without any survival label. Four
pieces carry the claim. A multi-source invariant encoder maps the observation to a
state through one shared observation map, so no single resource owns a coordinate.
A selective state-space operator advances that state along the axis, choosing how
much new observation to write and how much of the existing state to keep, on an
exponential-trapezoidal discretisation with a complex-valued diagonal spectrum and
a multi-input multi-output form, at a cost linear in the axis length. Three
label-free terms — masked multi-source reconstruction, cross-resource alignment and
trajectory consistency — train the representation. A depth-one read-out over the
pooled trajectory is the only place a survival label enters.

The repository ships that machinery, the theory the manuscript relies on
(identification up to an invertible linear transformation and the invariance of
discrimination), the resource readers for the four public archives, the eighteen
comparator rows of the main comparison under one shared protocol, the mechanism
evidence triad, the strata and modality-dropout arms, the label-efficiency arm, the
statistics the tables are reported with, and a two-pass verification layer that
writes the three root artefacts.

## What the manuscript reports and what this environment produced

The manuscript's cohorts are four public archives that are not staged here, and its
printed tables were calibrated on the manuscript's own cohort assembly. The
mechanisms run on cohorts generated from the manuscript's own state-space law, so
the code is executable and every claim can be checked against an independent
recomputation, but the printed cohort-level values are not reproducible without the
downloads.

| Reported quantity | Source | Status in this release |
| --- | --- | --- |
| Table 1, main comparison on the held-out split of the training resource | manuscript | `NOT_RUN`: the printed values come from a non-distributable cohort assembly; the same protocol is executable and its rows are written to the run artefacts |
| Table 2, component ablation | manuscript | executable; the caption's own arithmetic is internally inconsistent and is reported as such |
| Table 3, strata and modality dropout | manuscript | executable on generated cohorts |
| Table 4, state properties | manuscript | executable, including the axis-length and state-dimension sweeps and the shuffled ordering |
| Table 5, mechanism triad | manuscript | probe and perturbation arms executable; the necessity arm is the matched-capacity substitution |
| Table 6, external validation | manuscript | `BLOCKED`: the three external cohorts are served by archives that are not staged locally |
| Sec. 4.10, transfer tax | manuscript | executable; the arithmetic reproduces the printed range and reduction share |

The per-check outcome, including the checks that failed and the checks that could
not run, is in `verification_report.json`; `claim_to_code.json` carries the
paper-claim-to-code mapping with every intentional departure; a plain-text summary
is in `verification_summary.txt`. The shipped report holds 102 checks: 96 pass,
3 fail, 2 did not run and 1 is blocked.

The three failures are findings rather than defects, and each is recorded with its
measured values:

- `mech-01`: resource identity stays readable from the learned state at the
  training budget this environment runs (raw 0.643 against learned 0.752), so the
  invariance level the manuscript reaches is not reached here. The companion
  `mech-04` sweep does hold: raising the alignment weight moves the reading in the
  direction the manuscript describes.
- `mech-02`: the response grows with the size of the disturbance at the early end
  of the axis (-0.165 to -0.294) but not at the late end (-0.153 to -0.141), so the
  monotone-in-scale statement holds only at one end at this budget.
- `reported-02`: the manuscript's caption for the ablation states a smallest drop
  of 0.003 while its own table contains a drop of 0.002, so the ratio the caption
  prints (13.7x) does not follow from its table (20.5x). The inconsistency is
  recorded rather than reconciled.

The two checks that did not run are `table1-01` and `external-01`, both because the
manuscript's cohort assembly is not distributable; `table6-01` is blocked because
the three external cohorts are served by archives that are not staged here. Each of
those three carries the reason in its `detail` field.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

```bash
conda env create -f environment.yml
conda activate twindyn
```

```bash
docker build -t twindyn .
docker run --rm -v "$PWD/data:/data" twindyn
```

## Data

The readers never fetch anything. Each one inspects a local resource root and
reports which artefacts it can see; a resource whose files are absent is recorded
so its cohort-level values are reported as not run rather than substituted. Stage
the archives under `data/` in the layout the readers expect and run
`scripts/prepare_data.sh` to write the preparation manifest.

| Resource | Role | Access | Declared terms | Expected on-disk layout |
| --- | --- | --- | --- | --- |
| [TCGA-KIRC](https://portal.gdc.cancer.gov/projects/TCGA-KIRC) | training | GDC open tier, no application step | none declared by the source | `data/tcga_kirc/` with a GDC sample sheet, a clinical table and per-sample expression files |
| [CPTAC-ccRCC, PDC000127](https://pdc.cancer.gov/pdc/study/PDC000127) | external | open download | imaging collection CC BY 4.0; proteomic arm declares none | `data/cptac_ccrcc/` with a PDC proteome table and a biospecimen table |
| [GSE29609](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE29609) | external | freely downloadable | none declared by the source | `data/gse29609/` with the series matrix file, gzipped or plain |
| [E-MTAB-1980](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-1980) | external | openly released since 2013-10-16 | none declared by the source | `data/emtab1980/` with the SDRF file and the processed data matrix |

The manuscript's diagnostic whole-slide images come from The Cancer Imaging
Archive; that host is not reachable from the environment where this release was
assembled, so its collection page could not be re-checked and no imaging data is
bundled. `dataset_urls.txt` lists every link that was fetched and whose returned
content matched the resource the manuscript names, and records the endpoints that
could not be re-fetched with the reason.

Preprocessing the readers perform: the shared feature space is the **intersection**
of the observable features across the four resources, missing features are left
absent rather than imputed, and the compartment fractions come from a
reference-based non-negative least squares deconvolution whose configuration is
fixed before any outcome is seen. The compartment ordering that builds the axis is
in `src/twindyn/data/compartments.py` and is written to `ordered_axis.txt` by
`scripts/prepare_data.sh`.

## Running the reported experiments

One command per reported experiment. Configuration lives under `configs/`, composes
`model/base.yaml`, `data/base.yaml` and `train/base.yaml`, and accepts `--set`
overrides.

```bash
# primary result
python3 -m twindyn.cli.train --experiment main

# every ablation row of Table 2
bash scripts/launch_ablation.sh

# state properties and the internal-validation selection, Table 4 and Sec. 3.10
bash scripts/launch_sweep.sh --quantity tau_max --quantity state_dim --quantity mask_ratio

# the label-efficiency arm, Sec. 4.10
python3 -m twindyn.cli.train --experiment supplementary_label_efficiency

# strata and modality dropout, Table 3
python3 -m twindyn.cli.train --experiment supplementary_strata_and_dropout

# the mechanism triad, Table 5
python3 -m twindyn.cli.train --experiment supplementary_mechanism

# score a fitted initialisation on the hold-out split and the external cohorts
bash scripts/launch_eval.sh runs/main/checkpoints/initialisation_00.pt
```

Each run writes `run.json`, `tables.json` and `initialisations.json` under its
output directory, alongside one state file per indexed initialisation so the
read-out variants can reuse a representation that was learned once.

Reported configuration, taken from the manuscript: `tau_max = 16` and
`state_dim = 64` (Table 4), ten indexed initialisations, 1,000 bootstrap resamples
with 95% intervals, Holm-Bonferroni correction, the prespecified external
threshold of 0.03 concordance units and the decision margin of 0.005. The
quantities the manuscript leaves open — the masking ratio, the three objective
weights, the optimiser, the learning rate, the batch size, the epoch counts and the
deconvolution configuration — are engineering defaults in the configuration files;
`src/twindyn/evaluation/selection.py` implements the internal-validation procedure
the manuscript describes for choosing them. `claim_to_code.json` lists them under
`unreported_quantities`.

## Compute budget

The manuscript does not report a compute target. `configs/experiment/main.yaml`
asks for ten initialisations of a 3.2-million-parameter model at `tau_max = 16` and
`state_dim = 64` over 60 representation epochs and 40 read-out epochs; the analytic
scan cost is in `scan_cost` and is linear in the axis length with a quadratic
dependence on the state dimensionality, so the whole sweep fits on one accelerator
and also runs on CPU. This repository was assembled and verified on CPU; the smoke
configuration finishes in about a minute, and the full configuration is bounded by
the cohort size rather than by the operator.

## Verification

```bash
bash scripts/run_verification.sh
```

Pass one binds each claim in `src/twindyn/claims.py` to the files that carry it and
to the checks that test it. Pass two executes the pipeline and tests each claim
against an independent recomputation: a brute-force pairwise loop for the
concordance index, a dense matrix-exponential evaluation of the same recurrence for
the selective scan, `scipy.optimize.nnls` for the deconvolution, plain set
arithmetic for the shared feature space, and hand-computed values for the Cox
partial likelihood, the trajectory contrast and the Holm step-down. The pass also
covers data reading, the forward pass, every loss term, the backward pass on every
parameter group, a parameter update, a checkpoint round trip, a single-batch
overfit and a short two-stage training loop.

Results are reported as `PASS`, `FAIL`, `NOT_RUN` or `BLOCKED` from what ran, the
whole release is marked `PARTIALLY_VERIFIED` when anything is outstanding, and the
`integrity_manifest.json` is written last so it describes the finished tree. The
double verification found one real defect in this codebase: the survival objective
used a plain cumulative log-sum for the risk sets, which is wrong when follow-up
times are tied, and the hand-computed Breslow likelihood in the check exposed it.

## Repository layout

```text
src/twindyn/
  registry.py              study constants, paper values and the engineered defaults
  reported.py              the manuscript's printed numbers, transcribed with their anchors
  claims.py                the claim-to-code map of pass one
  verification.py          both passes and the artefact writers
  pipeline.py              one run: cohorts, splits, both stages, all arms, the tables
  data/                    compartments, axis, feature space, deconvolution, masking,
                           pairs, splits, schema, synthetic law, resource readers
  models/                  MSI, the selective scan and transition, the read-out,
                           the static control, the probe, the comparator family
  losses/                  equations (3)-(6) and the survival objective
  metrics/                 concordance, bootstrap, Holm-Bonferroni, the decision rule,
                           strata, label efficiency, transfer tax
  training/                optimiser, schedules, precision, EMA, checkpoints, seeding, engine
  evaluation/              views, selection, external arm, mechanism, ablation, reporting
  utils/                   configuration, atomic writers, logging, seeding
  cli/                     train, evaluate, ablation, sweep, prepare-data, verify
configs/                   model, data, train and one file per reported experiment
scripts/                   launch and preparation shells
tests/                     unit and integration suites plus the smoke and verification tests
```

## Tooling

```bash
python3 -m pytest -q
ruff check . && ruff format --check .
mypy src/twindyn
```

All three run clean. `mypy` is configured with `disallow_untyped_defs`, and the
overrides listed in `pyproject.toml` relax it only for the modules that build and
read the report payloads as nested plain mappings, where the outstanding annotation
work is the payload plumbing rather than the science. Every algorithm module — the
data layer, the models, the losses, the metrics, the training stack, the evaluation
views and selection, and the utilities — type-checks without an override.

Configuration is composed by `twindyn.utils.config`, which accepts `KEY=VALUE`
overrides on every command. Logging goes through the `logging` module; the writers
are atomic; the seeding utility is applied at the start of every run and the seed
travels with each written state file.

Third-party datasets, packages and the container base image, with their licences
and sources, are listed in `THIRD_PARTY_NOTICES.txt`.
