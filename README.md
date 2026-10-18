# SpeechPreserve

Implementation of **Speech-Preservation Routing Under Streaming Resource Constraints for Home Smart Displays**.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Generate an analytic voiced-modulation task with independent colored noise. Role NPZ files contain 16-kHz, 160-sample-hop noisy `x`, clean `y` and globally disjoint utterance `group` IDs. These are speech-like harmonic signals, not recorded/intelligible speech. `analytic_backends.json` selects FIR3/9/17 and an abstract operation clock; deadline/work figures describe that model. For host timing, use the separate profile → fit-risk → execute route with `backends.json` on the intended host.

```bash
python -m speech_routing.synthetic --output data/voiced_modulation --seed 13
```

## Run and evaluate

```bash
python -m speech_routing.runtime fit-risk --train data/voiced_modulation/train.npz --calibration data/voiced_modulation/calibration.npz --config analytic_backends.json --output runs/analytic-risk.pt --epochs 100 --seed 13
speech-routing execute --input data/voiced_modulation/test.npz --config analytic_backends.json --model runs/analytic-risk.pt --output runs/analytic-execution --variant full
```

## Ablations

Supported controls: `no_preservation`, `no_debt`, `no_switch_cost`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
speech-routing execute --input data/voiced_modulation/test.npz --config analytic_backends.json --model runs/analytic-risk.pt --output outputs/no_preservation --variant no_preservation
```

## Outputs

The generator writes role NPZs and source/setting metadata. `runs/analytic-risk.pt` contains fitted risk/calibration state. Execution writes actual waveform/action NPZ, call ledger and metrics. `completion_time_basis=analytic_operations` identifies abstract-clock deadlines separately from host timings.

Keep `analytic_backends.json` and the generated `setting.json` with this run; they define the selected backends, operation clock and source seed. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training or full evaluation was run. The GitHub Actions workflow checks source syntax and package installation.
