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

## Execute frozen enhancement backends on waveforms

The original `train/evaluate` path is the compact FIR setting. The separate
`execute` path runs the selected enhancement backend on each noisy hop. Activation
resets the chosen backend and replays at most `replay_frames` past noisy hops;
optional cross-fading executes both the old and new branches on the current hop.
Only past/current noisy samples enter routing. Clean speech is read afterwards for
scoring. Call ledgers include warm-up and cross-fade calls, actual host elapsed time,
backend-reported operation counts, and declared memory accounting.

A backend configuration contains `sample_rate`, `hop_ms`, `deadline_ms`,
`algorithmic_delay_ms`, `compute_budget_units`, `memory_budget_kib`, `damage_limit`,
`replay_frames`, `crossfade_samples`, and a `backends` array. Each entry has a unique
`name` and initial measured `latency_ms`, `work_units`, `workspace_kib`. For a real
causal FIR entry add `kind: "fir"` and `taps: [0.25, 0.5, 0.25]`. For an external
frozen enhancement model add `module: "/absolute/path/backend.py"` and its
`parameters` object. The local module exports
`create_backend(parameters, sample_rate)`, returning an object with `reset()`,
`resident_memory_kib`, and `process(frame)`; the latter returns a dictionary with
`waveform` (finite, one aligned hop), `work_units` (actual work performed), and
`workspace_kib` (measured temporary activation footprint). Device backends must
synchronize before returning so host timing includes completion. Sample alignment
and any fixed algorithmic delay belong to the declared backend contract. Existing
weights are supplied locally; this repository does not download or impersonate
RNNoise/DCCRN/other named enhancement models.

Fit a backend-specific risk head from actual executed frozen-backend transitions:

```bash
python -m speech_routing.runtime fit-risk --train data/prepared/train.npz --calibration data/prepared/calibration.npz --config backends.json --output runs/backend-risk.pt
speech-routing execute --input data/prepared/test.npz --config backends.json --model runs/backend-risk.pt --output runs/executed --variant full
```

`fit-risk` uses clean targets only on training/calibration groups, fits quality and
amplitude-sensitive damage on the waveform after actual activation/replay/fade,
and calibrates action-wise upper residual margins using held-out utterance maxima.
Feature normalization is fitted on training groups. Test groups must be excluded.
Too few calibration groups give infinite risk bounds and force the safe identity
fallback. A local `scorer_module` exporting `score(frame, context, action_names)`
can instead supply `quality` and nonnegative calibrated `damage_bound` vectors;
its calibration and group separation are the caller's responsibility.

Runtime variants are `full`, `no_preservation`, `no_debt`, and `no_switch_cost`;
measured replay/fade costs are charged even when their utility term is ablated.
`executed.npz` stores actual waveforms/actions; `execution.json` retains every call.
`backend_processing_p95_ms` times backend execution, while `completion_p95_ms`
includes scorer time and queued work. Deadline violation also includes declared
algorithmic delay. `accounted_peak_memory_kib` sums backend resident/workspace
reports and explicit router frame/replay/output buffers; overlapping reported
output buffers can be counted twice conservatively. It is a declared accounting
upper bound for these components, not a measured whole-process/device peak.
`memory_violation_accounted` uses that same convention. Work units are not energy.
Initial profiles must be measured on the target device, and completed calls update
only future decisions. No backend-risk training or device experiment has been run
as part of the logic audit.

## Selected executable FIR setting

`backends.json` fixes real causal moving-average FIR backends of length 3/9/17
at 16 kHz, 160-sample (10 ms) hops, 20 ms deadline, 1440 work units/hop, 256 KiB
accounted memory, two-hop replay, 16-sample cross-fade and damage limit 0.2.
This is the chosen frozen backend library; named neural models or downloaded
weights are unnecessary. The risk head remains fitted/calibrated to actual
waveforms emitted by these backends.

```bash
python -m speech_routing.runtime profile --input data/prepared/train.npz --config backends.json --output runs/profiled_backends.json
python -m speech_routing.runtime fit-risk --train data/prepared/train.npz --calibration data/prepared/calibration.npz --config runs/profiled_backends.json --output runs/backend-risk.pt --epochs 100 --seed 13
speech-routing execute --input data/prepared/test.npz --config runs/profiled_backends.json --model runs/backend-risk.pt --output runs/executed --variant full
```

Initial latency/workspace fields in backends.json are null and `profiled=false`;
routing rejects them until the profile command fills actual observed per-call
95th-percentile latency, work and workspace reports. Profiling uses noisy x only,
with one aligned hop per call and no labels. It does not establish a worst-case
hardware guarantee. Repeat profiling on the intended execution device/hop size.
The fixed 1 ms algorithmic_delay_ms is an allowed delay budget for this declared
causal FIR interface, not a measured device cost. Required inputs are real noisy/
clean speech with disjoint training/calibration/test utterance groups; at least
nine independent calibration groups per action are needed for a finite 90% bound.
No profile measurement or risk fitting was executed in this source-only update.
