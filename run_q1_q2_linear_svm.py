#!/usr/bin/env python3
"""
AwA2 Project - Question 1 & 2

Q1: Split each class into 60% training and 40% testing.
Q2: Train and evaluate a linear SVM on deep features.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from numpy.lib.format import open_memmap
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    feature_dir = (
        project_root
        / "AwA2"
        / "AwA2-features"
        / "Animals_with_Attributes2"
        / "Features"
        / "ResNet101"
    )
    base_dir = project_root / "AwA2" / "AwA2-base" / "Animals_with_Attributes2"

    parser = argparse.ArgumentParser(
        description="AwA2 Q1-Q2: class-wise split + linear SVM classification."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=feature_dir / "AwA2-features.txt",
        help="Path to AwA2 deep feature txt.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=feature_dir / "AwA2-labels.txt",
        help="Path to AwA2 labels txt (1-based class index).",
    )
    parser.add_argument(
        "--filenames",
        type=Path,
        default=feature_dir / "AwA2-filenames.txt",
        help="Path to image filename list.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=base_dir / "classes.txt",
        help="Path to class id/name mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs_q1_q2",
        help="Directory to save split/model/metrics.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Train ratio within each class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Max iterations for LinearSVC.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="LinearSVC C value (used when --no-cv).",
    )
    parser.add_argument(
        "--use-cv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use K-fold CV to select C.",
    )
    parser.add_argument(
        "--c-grid",
        type=str,
        default="0.01,0.1,1,10,100",
        help="Comma-separated C candidates for CV.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="K in K-fold CV.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Parallel jobs for GridSearchCV.",
    )
    parser.add_argument(
        "--cv-verbose",
        type=int,
        default=10,
        help="Verbosity level for GridSearchCV. Use >=10 for live fit logs.",
    )
    parser.add_argument(
        "--cache-npy",
        type=Path,
        default=feature_dir / "AwA2-features-float32.npy",
        help="Cache file for parsed features (float32 .npy).",
    )
    return parser.parse_args()


def parse_c_grid(c_grid: str) -> List[float]:
    values = [float(x.strip()) for x in c_grid.split(",") if x.strip()]
    if not values:
        raise ValueError("c-grid is empty.")
    return values


def load_labels(labels_path: Path) -> np.ndarray:
    labels = np.loadtxt(labels_path, dtype=np.int32)
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape={labels.shape}.")
    return labels


def load_filenames(path: Path, expected_n: int) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) != expected_n:
        raise ValueError(
            f"Filenames count mismatch: got {len(names)}, expected {expected_n}."
        )
    return names


def load_class_map(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cls_name = parts[1].replace("+", " ")
            mapping[cls_id] = cls_name
    return mapping


def infer_feature_dim(feature_txt: Path) -> int:
    with feature_txt.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    vec = np.fromstring(first_line, sep=" ", dtype=np.float32)
    if vec.size == 0:
        raise ValueError("Failed to parse first line of feature file.")
    return int(vec.size)


def build_feature_cache(
    feature_txt: Path,
    cache_npy: Path,
    n_samples: int,
    feature_dim: int,
) -> np.memmap:
    cache_npy.parent.mkdir(parents=True, exist_ok=True)
    mm = open_memmap(
        cache_npy,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, feature_dim),
    )

    with feature_txt.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=n_samples, desc="Parsing features")):
            vec = np.fromstring(line, sep=" ", dtype=np.float32)
            if vec.size != feature_dim:
                raise ValueError(
                    f"Feature dim mismatch at line {idx + 1}: "
                    f"got {vec.size}, expected {feature_dim}."
                )
            mm[idx, :] = vec

    if idx + 1 != n_samples:
        raise ValueError(
            f"Feature row count mismatch: got {idx + 1}, expected {n_samples}."
        )
    mm.flush()
    return mm


def load_features(feature_txt: Path, cache_npy: Path, n_samples: int) -> np.ndarray:
    if cache_npy.exists():
        x = np.load(cache_npy, mmap_mode="r")
        if x.shape[0] != n_samples:
            raise ValueError(
                f"Cached feature rows mismatch: {x.shape[0]} vs expected {n_samples}."
            )
        return x

    feature_dim = infer_feature_dim(feature_txt)
    print(
        f"Building feature cache at {cache_npy} "
        f"(n_samples={n_samples}, dim={feature_dim})..."
    )
    return build_feature_cache(feature_txt, cache_npy, n_samples, feature_dim)


def split_by_class(
    labels: np.ndarray, train_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}.")

    rng = np.random.default_rng(seed)
    train_indices: List[np.ndarray] = []
    test_indices: List[np.ndarray] = []

    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        n_train = int(np.floor(len(cls_idx) * train_ratio))
        n_train = max(1, min(n_train, len(cls_idx) - 1))
        train_indices.append(cls_idx[:n_train])
        test_indices.append(cls_idx[n_train:])

    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_map: Dict[int, str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cls in np.unique(y_true):
        mask = y_true == cls
        acc = float((y_pred[mask] == y_true[mask]).mean())
        name = class_map.get(int(cls), f"class_{int(cls)}")
        out[name] = acc
    return out


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        "Load data",
        "Split dataset",
        "Train SVM (CV)" if args.use_cv else "Train SVM",
        "Evaluate & save",
    ]
    stage_bar = tqdm(stages, desc="Overall progress", unit="step")

    # ── Stage 1: Load data ──
    stage_bar.set_postfix_str("Loading labels & features")
    y = load_labels(args.labels)
    n_samples = int(y.shape[0])
    tqdm.write(f"  Labels loaded: {n_samples} samples, {len(np.unique(y))} classes")

    _ = load_filenames(args.filenames, n_samples)
    class_map = load_class_map(args.classes)

    x = load_features(args.features, args.cache_npy, n_samples)
    tqdm.write(f"  Features ready: shape={x.shape}, dtype={x.dtype}")
    next(iter(stage_bar))

    # ── Stage 2: Split dataset ──
    stage_bar.set_postfix_str("60/40 class-wise split")
    train_idx, test_idx = split_by_class(y, args.train_ratio, args.seed)
    np.savez(
        args.output_dir / "split_indices.npz",
        train_idx=train_idx,
        test_idx=test_idx,
    )
    tqdm.write(f"  Train size={len(train_idx)}, Test size={len(test_idx)}")

    stage_bar.set_postfix_str("Copying to contiguous arrays")
    x_train = np.ascontiguousarray(x[train_idx])
    y_train = y[train_idx].copy()
    x_test = np.ascontiguousarray(x[test_idx])
    y_test = y[test_idx].copy()
    del x
    tqdm.write(f"  x_train={x_train.shape}, x_test={x_test.shape}")
    next(iter(stage_bar))

    # ── Stage 3: Train SVM ──
    if args.use_cv:
        c_values = parse_c_grid(args.c_grid)
        n_total_fits = len(c_values) * args.cv_folds
        stage_bar.set_postfix_str(
            f"GridSearchCV: {len(c_values)} C × {args.cv_folds} folds = {n_total_fits} fits"
        )

        cv = StratifiedKFold(
            n_splits=args.cv_folds, shuffle=True, random_state=args.seed
        )
        estimator = LinearSVC(
            dual=False,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid={"C": c_values},
            cv=cv,
            scoring="accuracy",
            n_jobs=args.n_jobs,
            pre_dispatch=args.n_jobs if args.n_jobs > 0 else "2*n_jobs",
            verbose=args.cv_verbose,
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_c = float(search.best_params_["C"])
        cv_best_score = float(search.best_score_)
        tqdm.write(f"  Best C={best_c}, CV accuracy={cv_best_score:.6f}")

        tqdm.write("  CV results per C:")
        for mean, std, params in zip(
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
            search.cv_results_["params"],
        ):
            tqdm.write(f"    C={params['C']:<8} accuracy={mean:.6f} ± {std:.6f}")
    else:
        stage_bar.set_postfix_str(f"Training with C={args.c}")
        model = LinearSVC(
            C=args.c,
            dual=False,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        model.fit(x_train, y_train)
        best_c = float(args.c)
        cv_best_score = None
        tqdm.write(f"  Trained with fixed C={best_c}")
    next(iter(stage_bar))

    # ── Stage 4: Evaluate & save ──
    stage_bar.set_postfix_str("Predicting & computing metrics")
    y_pred = model.predict(x_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    class_acc = per_class_accuracy(y_test, y_pred, class_map)

    model_path = args.output_dir / "linear_svm.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "n_samples_total": n_samples,
        "n_classes": int(len(np.unique(y))),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "use_cv": bool(args.use_cv),
        "selected_C": best_c,
        "cv_best_accuracy": cv_best_score,
        "test_accuracy": test_acc,
        "classification_report": report,
        "per_class_test_accuracy": class_acc,
        "elapsed_seconds": time.time() - t0,
    }
    metrics_path = args.output_dir / "metrics_q1_q2.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    next(iter(stage_bar))

    stage_bar.set_postfix_str("Done!")
    stage_bar.close()

    print("\n===== Final Result =====")
    print(f"Test Accuracy: {test_acc:.6f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Split indices saved to: {args.output_dir / 'split_indices.npz'}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
