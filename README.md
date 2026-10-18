# Speech-Preservation Routing Under Streaming Resource Constraints for Home Smart Displays

Implementation of **Speech-Preservation Routing Under Streaming Resource Constraints for Home Smart Displays**.

Three causal FIR enhancement actions share a preservation-risk predictor. Selection filters predicted amplitude-sensitive damage, hop/deadline latency, and resident-memory constraints, then trades predicted quality against compute debt and switching cost. Queue and compute-debt state evolve across frames of each sequence; bypass is the explicit fallback. Profile latency and memory must come from the intended device; the demo profile is a declared analytical scenario.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,T,F]` is noisy framed audio and `y[N,T,F]` clean desired speech. Keep speakers/utterances in one group. Supply action latency, switching latency, memory, hop, and deadline in config, in milliseconds/KiB as stated. Use DNS/VoiceBank-style mixtures with fixed sample-rate, framing, and target-speaker conventions.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
speech-routing demo --output data/example.npz --count 96 --config config.json
speech-routing prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
speech-routing train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
speech-routing evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
speech-routing plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
speech-routing calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `no_preservation`, `no_debt`, `no_switch_cost`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
speech-routing train --input data/prepared --output runs/baseline --config config.json --variant baseline
speech-routing evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Reproduction logic conventions

The action profile uses memory_kib as the total peak resident footprint for each executable action, including any shared library/state footprint. memory_budget_kib is the allowed peak, and switch_latency_ms adds a scalar activation delay whenever the chosen action changes, including transitions from bypass. Deadline metrics use completion time before the hop drains the queue. The included FIR action library is a compact algorithmic setting; reproducing the full paper enhancement comparisons requires the registered enhancement models and measured device profiles.
