"""Causal execution of local speech-enhancement backends with measured feedback."""
import importlib.util
import json
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import lfilter


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FIRBackend:
    """A real causal FIR backend for protocol checks and the compact setting."""
    def __init__(self, taps):
        self.taps = np.asarray(taps, dtype=float)
        if self.taps.ndim != 1 or not len(self.taps) or not np.isfinite(self.taps).all():
            raise ValueError("FIR taps must be a finite nonempty vector")
        self.resident_memory_kib = self.taps.nbytes / 1024
        self.reset()

    def reset(self):
        self.state = np.zeros(len(self.taps) - 1)

    def process(self, frame):
        waveform, self.state = lfilter(self.taps, [1.], frame, zi=self.state)
        return {"waveform": waveform, "work_units": len(frame) * len(self.taps),
                "workspace_kib": (waveform.nbytes + self.state.nbytes) / 1024}


class BackendExecutor:
    """Execute only chosen branches, including bounded replay and cross-fade."""
    def __init__(self, specifications, sample_rate, replay_frames=2, crossfade_samples=0):
        if not np.isfinite(sample_rate) or sample_rate <= 0 or replay_frames < 0 or crossfade_samples < 0:
            raise ValueError("Positive sample rate and nonnegative replay/crossfade lengths are required")
        self.backends = []
        for index, spec in enumerate(specifications):
            if spec.get("kind") == "fir":
                backend = FIRBackend(spec["taps"])
            else:
                module = load_module(spec["module"], "speech_backend_" + str(index))
                backend = module.create_backend(spec.get("parameters", {}), sample_rate)
            if not np.isfinite(backend.resident_memory_kib) or backend.resident_memory_kib < 0:
                raise ValueError("Backends must report nonnegative measured resident_memory_kib")
            self.backends.append(backend)
        self.resident_memory_kib = sum(b.resident_memory_kib for b in self.backends)
        self.history = deque(maxlen=replay_frames)
        self.crossfade_samples = crossfade_samples
        self.previous = -1
        for backend in self.backends:
            backend.reset()

    def _call(self, action, frame, warmup, calls):
        begin = perf_counter()
        result = self.backends[action].process(frame.copy())
        latency = (perf_counter() - begin) * 1000
        waveform = np.asarray(result["waveform"], dtype=float)
        work, workspace = float(result["work_units"]), float(result["workspace_kib"])
        if waveform.shape != frame.shape or not np.isfinite(waveform).all():
            raise ValueError("Backend output must be finite and aligned with the input hop")
        if not np.isfinite([work, workspace]).all() or min(work, workspace) < 0:
            raise ValueError("Backend work/activation memory must be nonnegative measured values")
        calls.append({"action": action, "warmup": warmup, "latency_ms": latency,
                      "work_units": work, "workspace_kib": workspace})
        return waveform

    def process(self, action, frame):
        frame = np.asarray(frame, dtype=float)
        if frame.ndim != 1 or not np.isfinite(frame).all() or not -1 <= action < len(self.backends):
            raise ValueError("A finite waveform hop and valid action are required")
        calls = []
        begin = perf_counter()
        changed = action != self.previous
        if action >= 0:
            if changed:
                self.backends[action].reset()
                for past in self.history:
                    self._call(action, past, True, calls)
            output = self._call(action, frame, False, calls)
        else:
            output = frame.copy()
        fade = min(self.crossfade_samples, len(frame)) if changed else 0
        if fade:
            old = self._call(self.previous, frame, False, calls) if self.previous >= 0 else frame
            blend = np.linspace(0, 1, fade)
            output[:fade] = (1 - blend) * old[:fade] + blend * output[:fade]
        router_bytes = 2 * frame.nbytes + output.nbytes + sum(past.nbytes for past in self.history)
        if fade and self.previous >= 0:
            router_bytes += old.nbytes
        self.history.append(frame.copy())
        self.previous = action
        work = sum(c["work_units"] for c in calls) + 2 * fade
        # Both old/new outputs coexist during a transition; include their workspaces.
        workspace = sum(sorted((c["workspace_kib"] for c in calls if not c["warmup"]), reverse=True)[:2])
        workspace = max(workspace, max((c["workspace_kib"] for c in calls), default=0.))
        return output, {"latency_ms": (perf_counter() - begin) * 1000, "work_units": work,
                        "accounted_peak_memory_kib": self.resident_memory_kib + workspace + router_bytes / 1024,
                        "replay_calls": sum(c["warmup"] for c in calls), "calls": calls,
                        "switched": changed, "crossfade_samples": fade}


def scorer(config, model_path, action_names):
    if config.get("scorer_module"):
        return load_module(config["scorer_module"], "speech_risk_scorer").score
    if model_path is None:
        raise ValueError("Supply a local scorer_module or a frozen risk checkpoint fitted to this backend library")
    import torch
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if state.get("backend_names") != action_names:
        raise ValueError("Use a backend-risk checkpoint fitted/calibrated on these executed backend_names")
    model = risk_network(len(action_names))
    model.load_state_dict(state["risk_state"])
    model.eval()
    margin = np.asarray(state["margin"])

    def score(frame, context, names):
        feature = torch.tensor([[frame.mean(), frame.std(), context["queue_ms"], context["debt"]]], dtype=torch.float32)
        with torch.no_grad():
            feature = (feature - torch.tensor(state["feature_mean"])) / torch.tensor(state["feature_scale"])
            raw = model(feature).reshape(2, len(names))
            return {"quality": raw[0].numpy(),
                    "damage_bound": torch.nn.functional.softplus(raw[1]).numpy() + margin}
    score.excluded_groups = state["train_groups"] + state["calibration_groups"]
    return score


def replay(frames, specifications, config, predict_risk):
    """Route a noisy sequence; clean targets are deliberately absent from this API."""
    if min(config["hop_ms"], config["deadline_ms"], config["compute_budget_units"]) <= 0:
        raise ValueError("Hop, deadline and compute budgets must be positive")
    if config.get("profiled") is False:
        raise ValueError("Run the profile command on the intended backend/device before routing")
    variant = config.get("variant", "full")
    if variant not in {"full", "no_preservation", "no_debt", "no_switch_cost"}:
        raise ValueError("Runtime variants: full, no_preservation, no_debt, no_switch_cost")
    names = [s["name"] for s in specifications]
    executor = BackendExecutor(specifications, config["sample_rate"], config.get("replay_frames", 2), config.get("crossfade_samples", 0))
    latency = np.array([s["latency_ms"] for s in specifications], dtype=float)
    work = np.array([s["work_units"] for s in specifications], dtype=float)
    workspace = np.array([s["workspace_kib"] for s in specifications], dtype=float)
    if not len(names) or len(set(names)) != len(names) or not np.isfinite(np.r_[latency, work, workspace]).all() or np.min(np.r_[latency, work, workspace]) < 0:
        raise ValueError("Unique backend names and nonnegative initial measurements are required")
    queue = debt = 0.
    outputs, actions, records = [], [], []
    for tick, frame in enumerate(np.asarray(frames, dtype=float)):
        context = {"queue_ms": queue, "debt": debt, "previous": executor.previous,
                   "history": np.array(executor.history), "sample_rate": config["sample_rate"]}
        score_begin = perf_counter()
        estimate = predict_risk(frame.copy(), context, names)
        score_ms = (perf_counter() - score_begin) * 1000
        quality, risk = [np.asarray(estimate[k], dtype=float) for k in ("quality", "damage_bound")]
        if quality.shape != (len(names),) or risk.shape != quality.shape or not np.isfinite(quality).all() or np.isnan(risk).any() or (risk < 0).any():
            raise ValueError("Risk scorer must return finite per-action quality and calibrated nonnegative damage_bound")
        changed = np.arange(len(names)) != executor.previous
        activation = changed * len(executor.history)
        old_latency = latency[executor.previous] if executor.previous >= 0 and config.get("crossfade_samples", 0) else 0.
        duration = score_ms + latency * (1 + activation) + changed * old_latency
        router_kib = ((4 if config.get("crossfade_samples", 0) and executor.previous >= 0 else 3) * frame.nbytes + sum(past.nbytes for past in executor.history)) / 1024
        memory = executor.resident_memory_kib + workspace + router_kib
        if config.get("crossfade_samples", 0) and executor.previous >= 0:
            memory += changed * workspace[executor.previous]
        feasible = (queue + duration <= config["hop_ms"]) & (config.get("algorithmic_delay_ms", 0.) + queue + duration <= config["deadline_ms"])
        feasible &= memory <= config["memory_budget_kib"]
        if config.get("variant", "full") != "no_preservation":
            feasible &= risk <= config["damage_limit"]
        debt_weight = 0. if variant == "no_debt" else config.get("debt_weight", .01)
        switch_weight = 0. if variant == "no_switch_cost" else config.get("switch_weight", .01)
        utility = quality - debt_weight * debt * work - switch_weight * changed
        action = int(np.argmax(np.where(feasible, utility, -np.inf))) if feasible.any() else -1
        old = executor.previous
        if old >= 0 and feasible[old] and action >= 0 and utility[action] - utility[old] <= config.get("hysteresis", 0.):
            action = old
        output, measured = executor.process(action, frame)
        completion = queue + score_ms + measured["latency_ms"]
        queue = max(0., completion - config["hop_ms"])
        debt = 0. if variant == "no_debt" else max(0., debt + measured["work_units"] - config["compute_budget_units"])
        # Completed calls update only future feasibility estimates.
        for call in measured["calls"]:
            i = call["action"]
            latency[i] = .8 * latency[i] + .2 * call["latency_ms"]
            work[i] = .8 * work[i] + .2 * call["work_units"]
            workspace[i] = max(workspace[i], call["workspace_kib"])
        outputs.append(output)
        actions.append(action)
        records.append({"tick": tick, "action": action, "name": names[action] if action >= 0 else "identity",
                        "damage_bound": [float(v) if np.isfinite(v) else None for v in risk], "feasible": feasible.tolist(), "queue_ms": queue,
                        "debt": debt, "completion_ms": completion, "scorer_ms": score_ms,
                        "deadline_violation": config.get("algorithmic_delay_ms", 0.) + completion > config["deadline_ms"],
                        "memory_violation_accounted": measured["accounted_peak_memory_kib"] > config["memory_budget_kib"], **measured})
    return np.stack(outputs), np.asarray(actions), records


def execute(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config["variant"] = args.variant
    specifications = config["backends"]
    names = [s["name"] for s in specifications]
    predict_risk = scorer(config, args.model, names)
    with np.load(args.input, allow_pickle=False) as archive:
        if getattr(predict_risk, "excluded_groups", []) and "group" not in archive:
            raise ValueError("Fitted-risk execution requires independent group IDs")
        if "group" in archive and np.intersect1d(archive["group"], getattr(predict_risk, "excluded_groups", [])).size:
            raise ValueError("Execution/evaluation groups overlap backend-risk fitting/calibration")
        frames = archive["x"]
        if frames.ndim != 3 or frames.shape[-1] != round(config['sample_rate']*config['hop_ms']/1000):
            raise ValueError("x must be sequences,frames,samples with the configured sample-rate/hop length")
        outputs, actions, all_records = [], [], []
        for sequence, waveform in enumerate(frames):
            output, action, records = replay(waveform, specifications, config, predict_risk)
            outputs.append(output)
            actions.append(action)
            all_records.extend(dict(row, sequence=sequence) for row in records)
        outputs, actions = np.stack(outputs), np.stack(actions)
        metrics = {"deadline_violation_actual": float(np.mean([r["deadline_violation"] for r in all_records])),
                   "memory_violation_accounted": float(np.mean([r["memory_violation_accounted"] for r in all_records])),
                   "backend_processing_p95_ms": float(np.quantile([r["latency_ms"] for r in all_records], .95)),
                   "completion_p95_ms": float(np.quantile([r["completion_ms"] for r in all_records], .95)),
                   "total_backend_calls": sum(len(r["calls"]) for r in all_records),
                   "total_replay_calls": sum(r["replay_calls"] for r in all_records),
                   "bypass_fraction": float(np.mean(actions < 0))}
        if "y" in archive:
            from .method import measure
            completion = np.array([r["completion_ms"] for r in all_records]).reshape(actions.shape) + config.get("algorithmic_delay_ms", 0.)
            metrics.update(measure({"prediction": outputs, "action": actions, "completion_ms": completion}, {"y": archive["y"]}, config))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "executed.npz", prediction=outputs, action=actions)
    (destination / "execution.json").write_text(json.dumps(all_records, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")


def risk_network(actions):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.SiLU(), torch.nn.Linear(32, 2 * actions))


def collect_risk_targets(data, specifications, config):
    """Measure frozen backend transitions on labeled training/calibration mixtures."""
    if data["x"].shape != data["y"].shape or data["x"].ndim != 3 or "group" not in data:
        raise ValueError("Backend-risk fitting requires aligned x/y sequences and independent group IDs")
    if data['x'].shape[-1] != round(config['sample_rate']*config['hop_ms']/1000):
        raise ValueError('Risk fitting hop length differs from sample_rate/hop_ms')
    features, actions, targets, groups = [], [], [], []
    for sequence, (noisy, clean) in enumerate(zip(data["x"], data["y"])):
        for old in range(-1, len(specifications)):
            for new in range(len(specifications)):
                engine = BackendExecutor(specifications, config["sample_rate"], config.get("replay_frames", 2), config.get("crossfade_samples", 0))
                queue = debt = 0.
                for tick, (frame, reference) in enumerate(zip(noisy, clean)):
                    action = old if tick % 2 == 0 else new
                    feature = [frame.mean(), frame.std(), queue, debt]
                    output, measured = engine.process(action, frame)
                    queue = max(0., queue + measured["latency_ms"] - config["hop_ms"])
                    debt = max(0., debt + measured["work_units"] - config["compute_budget_units"])
                    energy = float(reference @ reference)
                    if action < 0 or energy <= 1e-8:
                        continue
                    gain = float(output @ reference) / energy
                    residual = float(np.sum((output - gain * reference) ** 2))
                    damage = (gain - 1) ** 2 + config.get("damage_beta", 1.) * residual / energy
                    quality = 10 * np.log10(max(gain * gain * energy, 1e-8) / max(residual, 1e-8)) / 20
                    features.append(feature)
                    actions.append(action)
                    targets.append([quality, damage])
                    groups.append(data["group"][sequence])
    if not features:
        raise ValueError("No speech-active executed backend targets were available")
    return np.array(features, np.float32), np.array(actions), np.array(targets, np.float32), np.array(groups)


def fit_backend_risk(train, calibration, specifications, config, output, epochs=100, seed=13):
    """Fit risk on measured transitions; calibrate only on excluded utterance groups."""
    import torch
    if np.intersect1d(train["group"], calibration["group"]).size:
        raise ValueError("Backend-risk training and calibration groups overlap")
    torch.manual_seed(seed)
    tx, ta, ty, _ = collect_risk_targets(train, specifications, config)
    cx, ca, cy, groups = collect_risk_targets(calibration, specifications, config)
    mean, scale = tx.mean(0), tx.std(0).clip(1e-6)
    tx, cx = (tx - mean) / scale, (cx - mean) / scale
    count = len(specifications)
    model = risk_network(count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    x, action, target = torch.tensor(tx), torch.tensor(ta), torch.tensor(ty)
    for _ in range(epochs):
        for ids in torch.randperm(len(x)).split(128):
            raw = model(x[ids]).reshape(-1, 2, count)
            quality = raw[torch.arange(len(ids)), 0, action[ids]]
            damage = torch.nn.functional.softplus(raw[torch.arange(len(ids)), 1, action[ids]])
            loss = torch.nn.functional.huber_loss(quality, target[ids, 0]) + torch.nn.functional.huber_loss(damage, target[ids, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        raw = model(torch.tensor(cx)).reshape(-1, 2, count)
        damage = torch.nn.functional.softplus(raw[torch.arange(len(cx)), 1, torch.tensor(ca)]).numpy()
    margins = []
    for action in range(count):
        units = [max(0., float(np.max(cy[(groups == group) & (ca == action), 1] - damage[(groups == group) & (ca == action)])))
                 for group in np.unique(groups) if np.any((groups == group) & (ca == action))]
        rank = int(np.ceil((len(units) + 1) * (1 - config.get("risk_alpha", .1))))
        margins.append(float(np.sort(units)[rank - 1]) if 0 < rank <= len(units) else float("inf"))
    torch.save({"risk_state": model.state_dict(), "backend_names": [s["name"] for s in specifications],
                "margin": margins, "config": config, "feature_mean": mean.tolist(), "feature_scale": scale.tolist(), "train_groups": np.unique(train["group"]).tolist(),
                "calibration_groups": np.unique(calibration["group"]).tolist()}, output)



def profile_library(frames, specifications, config):
    """Measure a frozen library on noisy hops, without routing or clean targets."""
    import copy
    frames=np.asarray(frames,dtype=float)
    if frames.ndim!=3 or not frames.size or not np.isfinite(frames).all():
        raise ValueError('Profiling x must be finite sequences,frames,samples')
    if frames.shape[-1] != round(config['sample_rate']*config['hop_ms']/1000):
        raise ValueError('Profile hop length differs from sample_rate/hop_ms')
    measured=copy.deepcopy(config)
    measured['backends']=copy.deepcopy(specifications)
    for action in range(len(specifications)):
        calls=[]
        for sequence in frames:
            engine=BackendExecutor(specifications,config['sample_rate'],0,0)
            for frame in sequence:
                _,record=engine.process(action,frame)
                calls.extend(record['calls'])
        measured['backends'][action].update(
            latency_ms=float(np.quantile([row['latency_ms'] for row in calls],.95)),
            work_units=float(max(row['work_units'] for row in calls)),
            workspace_kib=float(max(row['workspace_kib'] for row in calls)))
    measured['profiled']=True
    measured['profile_scope']={'sequences':len(frames),'frames_per_sequence':frames.shape[1],
                               'samples_per_hop':frames.shape[2],'batch_size':1,
                               'latency_statistic':'observed_per_call_p95_ms',
                               'memory_statistic':'backend_report_plus_router_buffer_accounting'}
    return measured


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['fit-risk','profile'])
    parser.add_argument('--train')
    parser.add_argument('--calibration')
    parser.add_argument('--input')
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--seed',type=int,default=13)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='profile':
        if not args.input:parser.error('profile requires --input noisy x[N,T,F] NPZ')
        with np.load(args.input,allow_pickle=False) as archive:
            measured=profile_library(archive['x'],config['backends'],config)
        destination.write_text(json.dumps(measured,indent=2,allow_nan=False),encoding='utf-8')
    else:
        if not args.train or not args.calibration:parser.error('fit-risk requires --train and --calibration')
        with np.load(args.train,allow_pickle=False) as archive:
            train={key:archive[key] for key in archive.files}
        with np.load(args.calibration,allow_pickle=False) as archive:
            calibration={key:archive[key] for key in archive.files}
        fit_backend_risk(train,calibration,config['backends'],config,destination,args.epochs,args.seed)


if __name__=='__main__':
    main()
