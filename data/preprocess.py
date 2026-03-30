import numpy as np
from pathlib import Path


DEAP_CHANNELS = 32
DEAP_SR = 128
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
SEGMENT_SEC = 5
SEGMENT_SAMPLES = SEGMENT_SEC * DEAP_SR


def bandpower(signal: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    psd = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(np.mean(psd[mask])) if mask.any() else 0.0


def extract_eeg_features(eeg: np.ndarray, sr: int = DEAP_SR) -> np.ndarray:
    n_chan, n_time = eeg.shape
    alpha = np.mean([bandpower(eeg[c], sr, *BANDS["alpha"]) for c in range(n_chan)])
    theta = np.mean([bandpower(eeg[c], sr, *BANDS["theta"]) for c in range(n_chan)])
    beta  = np.mean([bandpower(eeg[c], sr, *BANDS["beta"])  for c in range(n_chan)])
    total = alpha + theta + beta + 1e-10
    return np.array([alpha / total, theta / total, beta / total], dtype=np.float32)


def eeg_features_to_state(features: np.ndarray) -> np.ndarray:
    alpha_r, theta_r, beta_r = features
    attention = float(np.clip(alpha_r * 1.8 - theta_r * 0.5, 0.0, 1.0))
    fatigue   = float(np.clip(theta_r * 2.0 - alpha_r * 0.3, 0.0, 1.0))
    stress    = float(np.clip(beta_r  * 2.5 - alpha_r * 0.4, 0.0, 1.0))
    return np.array([attention, fatigue, stress], dtype=np.float32)


def load_deap_subject(dat_path: str) -> dict:
    import pickle

    with open(dat_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    return {
        "data": data["data"].astype(np.float32),
        "labels": data["labels"].astype(np.float32),
    }


def build_trajectories_from_deap(data_dir: str, n_subjects: int = 5) -> list[dict]:
    data_dir = Path(data_dir)
    trajectories = []

    for subject_file in sorted(data_dir.glob("s*.dat"))[:n_subjects]:
        subj = load_deap_subject(str(subject_file))
        eeg_data = subj["data"][:, :DEAP_CHANNELS, :]
        labels   = subj["labels"]

        for trial_idx in range(eeg_data.shape[0]):
            trial_eeg = eeg_data[trial_idx]
            n_segments = trial_eeg.shape[1] // SEGMENT_SAMPLES

            feature_seq, state_seq = [], []
            for seg_i in range(n_segments):
                start = seg_i * SEGMENT_SAMPLES
                seg   = trial_eeg[:, start : start + SEGMENT_SAMPLES]
                feats = extract_eeg_features(seg)
                state = eeg_features_to_state(feats)
                feature_seq.append(feats)
                state_seq.append(state)

            if len(state_seq) >= 3:
                valence = labels[trial_idx, 0] / 9.0
                arousal = labels[trial_idx, 1] / 9.0
                trajectories.append({
                    "features":   np.stack(feature_seq).astype(np.float32),
                    "states":     np.stack(state_seq).astype(np.float32),
                    "valence":    float(valence),
                    "arousal":    float(arousal),
                    "subject":    subject_file.stem,
                    "trial":      int(trial_idx),
                })

    return trajectories


def normalize_trajectories(trajectories: list[dict]) -> tuple[list[dict], dict]:
    all_states = np.concatenate([t["states"] for t in trajectories], axis=0)
    mean = all_states.mean(axis=0)
    std  = all_states.std(axis=0) + 1e-8

    normalized = []
    for t in trajectories:
        normalized.append({
            **t,
            "states_norm": ((t["states"] - mean) / std).astype(np.float32),
        })

    return normalized, {"mean": mean.tolist(), "std": std.tolist()}


def build_synthetic_deap_trajectories(n_subjects: int = 8, n_trials: int = 10) -> list[dict]:
    rng = np.random.default_rng(42)
    trajectories = []

    for subj_i in range(n_subjects):
        baseline_alpha = rng.uniform(0.35, 0.55)
        baseline_theta = rng.uniform(0.20, 0.35)
        baseline_beta  = rng.uniform(0.15, 0.30)

        for trial_i in range(n_trials):
            n_seg = rng.integers(8, 16)
            feature_seq, state_seq = [], []

            alpha_r = baseline_alpha + rng.normal(0, 0.02)
            theta_r = baseline_theta + rng.normal(0, 0.02)
            beta_r  = baseline_beta  + rng.normal(0, 0.02)

            for _ in range(n_seg):
                total = alpha_r + theta_r + beta_r + 1e-10
                feats = np.array([alpha_r / total, theta_r / total, beta_r / total], dtype=np.float32)
                state = eeg_features_to_state(feats)

                feature_seq.append(feats)
                state_seq.append(state)

                alpha_r += rng.normal(0, 0.015)
                theta_r += rng.normal(0, 0.015)
                beta_r  += rng.normal(0, 0.012)
                alpha_r  = float(np.clip(alpha_r, 0.05, 0.90))
                theta_r  = float(np.clip(theta_r, 0.05, 0.90))
                beta_r   = float(np.clip(beta_r,  0.05, 0.90))

            valence = float(np.clip(rng.uniform(0.3, 0.8), 0, 1))
            arousal = float(np.clip(rng.uniform(0.2, 0.9), 0, 1))

            trajectories.append({
                "features":   np.stack(feature_seq).astype(np.float32),
                "states":     np.stack(state_seq).astype(np.float32),
                "valence":    valence,
                "arousal":    arousal,
                "subject":    f"synth_s{subj_i:02d}",
                "trial":      trial_i,
            })

    return trajectories
