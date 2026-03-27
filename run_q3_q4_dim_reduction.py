#!/usr/bin/env python3
"""
AwA2 Project - Question 3 & 4

Q3: Reduce the dimensionality of deep learning features using three methods:
        1. Feature Selection  : SelectKBest with ANOVA F-test   (sklearn)
        2. Feature Projection : PCA                              (sklearn)
        3. Feature Learning   : Sparse Coding (OMP)              (sklearn)
    Perform LinearSVC classification at each (method, dim) pair and record
    how performance varies with the number of retained dimensions.

Q4: Identify the optimal dimensionality-reduction method and the optimal
    target dimensionality.

Speed strategy
--------------
By default the script reuses the best C found in Q2 (C=0.01) for every
LinearSVC fit, which reduces the SVM work from 25 fits per setting
(5-fold × 5 C values) down to exactly 1 fit per setting — a ~25x speedup.
Pass --use-cv to re-enable full cross-validation if desired.

Usage
-----
# Fast run, all three methods, fixed C from Q2 (default):
    python run_q3_q4_dim_reduction.py

# Same but skip sparse coding:
    python run_q3_q4_dim_reduction.py --no-sparse-coding

# Re-enable K-fold CV for C selection:
    python run_q3_q4_dim_reduction.py --use-cv

# Override the fixed C value:
    python run_q3_q4_dim_reduction.py --fixed-c 0.1

# Custom dimension lists (comma-separated):
    python run_q3_q4_dim_reduction.py --sel-dims 64,256,1024 --pca-dims 64,256,1024

Notes
-----
* All three pipelines share the *same* 60/40 class-wise split produced in Q1/Q2
  (loaded from split_indices.npz) so results are directly comparable.
* A single StandardScaler is fitted on the *training* split and applied to both
  train and test before dimensionality reduction, ensuring fair comparison.
* The third method uses fast sparse coding with OMP and a fixed dictionary
  initialised from PCA components or random training samples.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import sparse_encode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── Default experiment settings ───────────────────────────────────────────────
DEFAULT_SEL_DIMS: List[int] = [32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 2048]
DEFAULT_PCA_DIMS: List[int] = [32, 64, 128, 256, 512, 768, 1024, 1536]
DEFAULT_SC_DIMS: List[int] = [32, 64, 128, 256, 512, 1024]

# C value carried over from Q2 cross-validation
Q2_BEST_C: float = 0.01

# CV grid (only used when --use-cv is passed)
CV_C_GRID: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0]
CV_FOLDS: int = 5

MAX_ITER: int = 5000


# ═══════════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════


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

    p = argparse.ArgumentParser(
        description="AwA2 Q3-Q4: Dimensionality reduction + LinearSVC evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ──────────────────────────────────────────────────────────
    p.add_argument("--features", type=Path, default=feature_dir / "AwA2-features.txt")
    p.add_argument("--labels", type=Path, default=feature_dir / "AwA2-labels.txt")
    p.add_argument(
        "--cache-npy",
        type=Path,
        default=feature_dir / "AwA2-features-float32.npy",
        help="float32 .npy cache built by run_q1_q2_linear_svm.py.",
    )
    p.add_argument(
        "--split-npz",
        type=Path,
        default=project_root / "outputs_q1_q2" / "split_indices.npz",
        help="Split indices saved by run_q1_q2_linear_svm.py.",
    )
    p.add_argument("--output-dir", type=Path, default=project_root / "outputs_q3_q4")

    # ── Dimension lists ──────────────────────────────────────────────────────
    p.add_argument(
        "--sel-dims",
        type=str,
        default=",".join(map(str, DEFAULT_SEL_DIMS)),
        help="Target dims for SelectKBest (comma-separated).",
    )
    p.add_argument(
        "--pca-dims",
        type=str,
        default=",".join(map(str, DEFAULT_PCA_DIMS)),
        help="Target dims for PCA (comma-separated).",
    )
    p.add_argument(
        "--sc-dims",
        type=str,
        default=",".join(map(str, DEFAULT_SC_DIMS)),
        help="Code dimensions for sparse coding (comma-separated).",
    )

    # ── Method toggles ───────────────────────────────────────────────────────
    p.add_argument(
        "--no-selection",
        action="store_true",
        help="Skip feature-selection experiments.",
    )
    p.add_argument("--no-pca", action="store_true", help="Skip PCA experiments.")
    p.add_argument(
        "--no-sparse-coding",
        action="store_true",
        help="Skip sparse-coding experiments.",
    )

    # ── SVM / C settings ─────────────────────────────────────────────────────
    p.add_argument(
        "--fixed-c",
        type=float,
        default=Q2_BEST_C,
        help="Fixed C for LinearSVC (used unless --use-cv is set).",
    )
    p.add_argument(
        "--use-cv",
        action="store_true",
        help="Re-enable K-fold CV to select C for every setting. "
        "Much slower (~25x) but more rigorous.",
    )
    p.add_argument(
        "--c-grid",
        type=str,
        default=",".join(map(str, CV_C_GRID)),
        help="C candidates for GridSearchCV (only when --use-cv).",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS,
        help="K for stratified K-fold CV (only when --use-cv).",
    )
    p.add_argument(
        "--max-iter", type=int, default=MAX_ITER, help="Max iterations for LinearSVC."
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV (-1 = all cores).",
    )

    # ── Sparse coding settings ───────────────────────────────────────────────
    p.add_argument(
        "--sc-dict-init",
        type=str,
        choices=["pca", "random"],
        default="pca",
        help="Dictionary initialisation for sparse coding.",
    )
    p.add_argument(
        "--sc-omp-k",
        type=int,
        default=10,
        help="Target sparsity (number of non-zeros) for OMP sparse codes.",
    )
    p.add_argument(
        "--sc-transform-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for sparse_encode transform.",
    )
    p.add_argument(
        "--sc-save-artifacts",
        action="store_true",
        help="Save dictionary matrices (.npy) for each sparse-coding dim.",
    )
    p.add_argument(
        "--merge-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge with existing metrics_q3_q4.json to avoid overwriting other methods.",
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_labels(path: Path) -> np.ndarray:
    labels = np.loadtxt(path, dtype=np.int32)
    assert labels.ndim == 1, f"Expected 1-D labels, got shape {labels.shape}."
    return labels


def load_features(cache_npy: Path, feature_txt: Path, n_samples: int) -> np.ndarray:
    """Return float32 feature matrix; prefers the .npy cache."""
    if cache_npy.exists():
        x = np.load(cache_npy, mmap_mode="r")
        assert x.shape[0] == n_samples, (
            f"Cache rows {x.shape[0]} != expected {n_samples}."
        )
        print(f"  Loaded cache: shape={x.shape}, dtype={x.dtype}")
        return x

    print(f"  Cache not found – parsing {feature_txt}")
    with feature_txt.open("r") as f:
        first = f.readline()
    dim = len(first.split())
    x = np.zeros((n_samples, dim), dtype=np.float32)
    with feature_txt.open("r") as f:
        for i, line in enumerate(tqdm(f, total=n_samples, desc="Parsing")):
            x[i] = np.fromstring(line, sep=" ", dtype=np.float32)
    print(f"  Parsed: shape={x.shape}")
    return x


def load_split(split_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(split_npz)
    return data["train_idx"], data["test_idx"]


# ═══════════════════════════════════════════════════════════════════════════════
#  SVM helpers  (fast fixed-C path  +  optional CV path)
# ═══════════════════════════════════════════════════════════════════════════════


def fit_svm_fixed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    C: float,
    max_iter: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Train a single LinearSVC with the given fixed C and evaluate on the test set.
    Returns a result dict with the same keys as fit_svm_cv so downstream code
    is identical for both paths.
    """
    model = LinearSVC(C=C, dual=False, max_iter=max_iter, random_state=seed)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {
        "best_C": C,
        "cv_accuracy": None,  # no CV was run
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "cv_results": [],  # empty — consistent schema
    }


def fit_svm_cv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: List[float],
    cv_folds: int,
    max_iter: int,
    n_jobs: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Select C via stratified K-fold CV, refit on the full training set, and
    evaluate on the test set.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        LinearSVC(dual=False, max_iter=max_iter, random_state=seed),
        param_grid={"C": c_grid},
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(x_train, y_train)
    y_pred = search.predict(x_test)
    return {
        "best_C": float(search.best_params_["C"]),
        "cv_accuracy": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "cv_results": [
            {"C": float(p["C"]), "mean": float(m), "std": float(s)}
            for p, m, s in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
            )
        ],
    }


def fit_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    use_cv: bool,
    fixed_c: float,
    c_grid: List[float],
    cv_folds: int,
    max_iter: int,
    n_jobs: int,
    seed: int,
) -> Dict[str, Any]:
    """Dispatcher: CV path or fast fixed-C path."""
    if use_cv:
        return fit_svm_cv(
            x_train,
            y_train,
            x_test,
            y_test,
            c_grid=c_grid,
            cv_folds=cv_folds,
            max_iter=max_iter,
            n_jobs=n_jobs,
            seed=seed,
        )
    return fit_svm_fixed(
        x_train,
        y_train,
        x_test,
        y_test,
        C=fixed_c,
        max_iter=max_iter,
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Method 1 – Feature Selection: SelectKBest (ANOVA F-test)
# ═══════════════════════════════════════════════════════════════════════════════


def run_feature_selection(
    x_train_scaled: np.ndarray,
    y_train: np.ndarray,
    x_test_scaled: np.ndarray,
    y_test: np.ndarray,
    dims: List[int],
    input_dim: int,
    *,
    use_cv: bool,
    fixed_c: float,
    c_grid: List[float],
    cv_folds: int,
    max_iter: int,
    n_jobs: int,
    seed: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    SelectKBest with ANOVA F-statistics.

    F-scores are computed once; for each target k we simply slice the top-k
    indices – equivalent to running SelectKBest(k=k).fit() independently but
    much faster.
    """
    print("\n" + "═" * 60)
    print("  Method 1 — Feature Selection (SelectKBest / ANOVA F-test)")
    print("═" * 60)

    results: List[Dict[str, Any]] = []

    # Compute F-scores once
    print("  Computing ANOVA F-statistics … ", end="", flush=True)
    t0 = time.time()
    selector_full = SelectKBest(f_classif, k="all")
    selector_full.fit(x_train_scaled, y_train)
    f_scores = selector_full.scores_
    print(f"done  ({time.time() - t0:.1f}s)")

    np.save(output_dir / "selection_f_scores.npy", f_scores.astype(np.float32))

    sorted_indices = np.argsort(f_scores)[::-1]  # descending F-score

    for k in tqdm(dims, desc="  SelectKBest dims"):
        k_actual = min(k, input_dim)
        top_k_idx = np.sort(sorted_indices[:k_actual])

        x_tr_k = x_train_scaled[:, top_k_idx]
        x_te_k = x_test_scaled[:, top_k_idx]

        t_svm = time.time()
        svm_res = fit_svm(
            x_tr_k,
            y_train,
            x_te_k,
            y_test,
            use_cv=use_cv,
            fixed_c=fixed_c,
            c_grid=c_grid,
            cv_folds=cv_folds,
            max_iter=max_iter,
            n_jobs=n_jobs,
            seed=seed,
        )
        elapsed_svm = time.time() - t_svm

        record = {
            "method": "selection",
            "dim": k_actual,
            "elapsed_dim_reduction_s": 0.0,
            "elapsed_svm_s": elapsed_svm,
            **svm_res,
        }
        results.append(record)
        tqdm.write(
            f"    k={k_actual:>5d} | test_acc={svm_res['test_accuracy']:.4f} "
            f"| C={svm_res['best_C']:.4g} "
            f"| svm_time={elapsed_svm:.1f}s"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Method 2 – Feature Projection: PCA
# ═══════════════════════════════════════════════════════════════════════════════


def run_pca(
    x_train_scaled: np.ndarray,
    y_train: np.ndarray,
    x_test_scaled: np.ndarray,
    y_test: np.ndarray,
    dims: List[int],
    input_dim: int,
    *,
    use_cv: bool,
    fixed_c: float,
    c_grid: List[float],
    cv_folds: int,
    max_iter: int,
    n_jobs: int,
    seed: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    PCA projection.

    We fit PCA(n_components=max(dims)) once, obtaining all required principal
    components in a single SVD pass.  For each target k we slice the first k
    columns – mathematically identical to fitting PCA(k) independently.
    """
    print("\n" + "═" * 60)
    print("  Method 2 — Feature Projection (PCA)")
    print("═" * 60)

    results: List[Dict[str, Any]] = []
    max_dim = min(max(dims), input_dim)

    print(
        f"  Fitting PCA(n_components={max_dim}) on training set … ",
        end="",
        flush=True,
    )
    t0 = time.time()
    pca = PCA(n_components=max_dim, random_state=seed)
    x_train_pca_full = pca.fit_transform(x_train_scaled)
    x_test_pca_full = pca.transform(x_test_scaled)
    elapsed_pca = time.time() - t0
    print(f"done  ({elapsed_pca:.1f}s)")

    np.save(
        output_dir / "pca_explained_variance_ratio.npy",
        pca.explained_variance_ratio_.astype(np.float32),
    )

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"  Cumulative explained variance @ {max_dim} PCs: {cumvar[-1]:.4f}")
    for k in [32, 64, 128, 256, 512]:
        if k <= max_dim:
            print(f"    top-{k:4d} PCs → {cumvar[k - 1]:.4f}")

    for k in tqdm(dims, desc="  PCA dims"):
        k_actual = min(k, max_dim)
        x_tr_k = x_train_pca_full[:, :k_actual]
        x_te_k = x_test_pca_full[:, :k_actual]

        t_svm = time.time()
        svm_res = fit_svm(
            x_tr_k,
            y_train,
            x_te_k,
            y_test,
            use_cv=use_cv,
            fixed_c=fixed_c,
            c_grid=c_grid,
            cv_folds=cv_folds,
            max_iter=max_iter,
            n_jobs=n_jobs,
            seed=seed,
        )
        elapsed_svm = time.time() - t_svm

        record = {
            "method": "pca",
            "dim": k_actual,
            "explained_variance_cumulative": float(cumvar[k_actual - 1]),
            "elapsed_dim_reduction_s": elapsed_pca / max_dim * k_actual,
            "elapsed_svm_s": elapsed_svm,
            **svm_res,
        }
        results.append(record)
        tqdm.write(
            f"    k={k_actual:>5d} | test_acc={svm_res['test_accuracy']:.4f} "
            f"| C={svm_res['best_C']:.4g} "
            f"| cum_var={cumvar[k_actual - 1]:.4f} "
            f"| svm_time={elapsed_svm:.1f}s"
        )

    joblib.dump(pca, output_dir / "pca_model.joblib")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Method 3 – Feature Learning: Sparse Coding (OMP)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_dictionary(
    x_train_scaled: np.ndarray,
    code_dim: int,
    *,
    dict_init: str,
    seed: int,
) -> np.ndarray:
    if dict_init == "pca":
        pca = PCA(n_components=code_dim, random_state=seed)
        pca.fit(x_train_scaled)
        dictionary = pca.components_.astype(np.float32)
    elif dict_init == "random":
        rng = np.random.default_rng(seed + code_dim)
        idx = rng.choice(x_train_scaled.shape[0], size=code_dim, replace=False)
        dictionary = x_train_scaled[idx].astype(np.float32).copy()
        norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
        dictionary = dictionary / np.maximum(norms, 1e-8)
    else:
        raise ValueError(f"Unsupported dict_init: {dict_init}")
    return dictionary


def run_sparse_coding(
    x_train_scaled: np.ndarray,
    y_train: np.ndarray,
    x_test_scaled: np.ndarray,
    y_test: np.ndarray,
    dims: List[int],
    *,
    use_cv: bool,
    fixed_c: float,
    c_grid: List[float],
    cv_folds: int,
    max_iter: int,
    n_jobs: int,
    seed: int,
    dict_init: str,
    omp_k: int,
    transform_n_jobs: int,
    save_artifacts: bool,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Build a fixed dictionary for each target code dimension, perform OMP sparse
    coding on train/test features, then evaluate LinearSVC on sparse codes.
    """
    print("\n" + "═" * 60)
    print("  Method 3 — Feature Learning (Sparse Coding + OMP)")
    print("═" * 60)

    sc_dir = output_dir / "sparse_coding_artifacts"
    sc_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for code_dim in dims:
        print(f"\n  ── Code dim = {code_dim} ──")

        t_sc = time.time()
        dictionary = _build_dictionary(
            x_train_scaled,
            code_dim=code_dim,
            dict_init=dict_init,
            seed=seed,
        )
        omp_k_actual = max(1, min(omp_k, code_dim))
        x_tr_enc = sparse_encode(
            x_train_scaled,
            dictionary,
            algorithm="omp",
            n_nonzero_coefs=omp_k_actual,
            n_jobs=transform_n_jobs,
        ).astype(np.float32)
        x_te_enc = sparse_encode(
            x_test_scaled,
            dictionary,
            algorithm="omp",
            n_nonzero_coefs=omp_k_actual,
            n_jobs=transform_n_jobs,
        ).astype(np.float32)
        elapsed_sc = time.time() - t_sc

        recon_train = np.matmul(x_tr_enc, dictionary)
        recon_test = np.matmul(x_te_enc, dictionary)
        train_recon_mse = float(np.mean((x_train_scaled - recon_train) ** 2))
        test_recon_mse = float(np.mean((x_test_scaled - recon_test) ** 2))
        train_code_density = float(np.mean(np.abs(x_tr_enc) > 1e-8))
        print(
            f"  Encoded: train={x_tr_enc.shape}, test={x_te_enc.shape}  "
            f"(SC transform {elapsed_sc:.1f}s, train_mse={train_recon_mse:.6f}, "
            f"code_density={train_code_density:.4f})"
        )

        if save_artifacts:
            np.save(sc_dir / f"dictionary_dim{code_dim}.npy", dictionary.astype(np.float32))

        t_svm = time.time()
        svm_res = fit_svm(
            x_tr_enc,
            y_train,
            x_te_enc,
            y_test,
            use_cv=use_cv,
            fixed_c=fixed_c,
            c_grid=c_grid,
            cv_folds=cv_folds,
            max_iter=max_iter,
            n_jobs=n_jobs,
            seed=seed,
        )
        elapsed_svm = time.time() - t_svm

        record = {
            "method": "sparse_coding",
            "dim": code_dim,
            "dictionary_init": dict_init,
            "omp_k": int(omp_k_actual),
            "train_recon_mse": train_recon_mse,
            "test_recon_mse": test_recon_mse,
            "train_code_density": train_code_density,
            "elapsed_dim_reduction_s": elapsed_sc,
            "elapsed_svm_s": elapsed_svm,
            **svm_res,
        }
        results.append(record)
        tqdm.write(
            f"    code_dim={code_dim:>5d} | test_acc={svm_res['test_accuracy']:.4f} "
            f"| C={svm_res['best_C']:.4g} "
            f"| sc_time={elapsed_sc:.1f}s | svm_time={elapsed_svm:.1f}s"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Q4: find optimal method & dimensionality
# ═══════════════════════════════════════════════════════════════════════════════


def find_optimal(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(all_results, key=lambda r: r["test_accuracy"])


def summarise_by_method(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Dict]:
    methods = sorted({r["method"] for r in all_results})
    summary: Dict[str, Dict] = {}
    for m in methods:
        subset = sorted(
            [r for r in all_results if r["method"] == m],
            key=lambda r: r["dim"],
        )
        best = max(subset, key=lambda r: r["test_accuracy"])
        summary[m] = {
            "best_dim": best["dim"],
            "best_test_accuracy": best["test_accuracy"],
            "best_cv_accuracy": best["cv_accuracy"],
            "best_C": best["best_C"],
            "all_dims": [r["dim"] for r in subset],
            "all_accuracies": [r["test_accuracy"] for r in subset],
            "all_cv_accuracies": [r["cv_accuracy"] for r in subset],
        }
    return summary


def merge_with_existing_results(
    output_path: Path,
    new_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not output_path.exists():
        return new_results
    with output_path.open("r", encoding="utf-8") as f:
        old = json.load(f)
    old_results = old.get("all_results", [])
    merged: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in old_results + new_results:
        merged[(r["method"], int(r["dim"]))] = r
    return sorted(merged.values(), key=lambda r: (r["method"], int(r["dim"])))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    t_global = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sel_dims = parse_int_list(args.sel_dims)
    pca_dims = parse_int_list(args.pca_dims)
    sc_dims = parse_int_list(args.sc_dims)
    c_grid = parse_float_list(args.c_grid)

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  AwA2 Q3/Q4 — Dimensionality Reduction Experiments")
    print("=" * 62)
    if args.use_cv:
        print(f"  C selection    : 5-fold CV  over {c_grid}")
    else:
        print(f"  C selection    : fixed C = {args.fixed_c}  (from Q2 best)")
    print(f"  Selection dims : {sel_dims}")
    print(f"  PCA dims       : {pca_dims}")
    print(f"  SC dims        : {sc_dims}")
    print(f"  SC dict init   : {args.sc_dict_init}")
    print(f"  SC OMP k       : {args.sc_omp_k}")
    print(f"  Output dir     : {args.output_dir}")
    print()

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/5] Loading labels …")
    y = load_labels(args.labels)
    n_samples = int(y.shape[0])

    print(f"[2/5] Loading features  (n_samples={n_samples}) …")
    x = load_features(args.cache_npy, args.features, n_samples)
    input_dim = int(x.shape[1])
    print(f"      Feature dim: {input_dim}")

    print("[3/5] Loading split indices …")
    if not args.split_npz.exists():
        raise FileNotFoundError(
            f"Split file not found: {args.split_npz}\n"
            "Please run run_q1_q2_linear_svm.py first."
        )
    train_idx, test_idx = load_split(args.split_npz)
    print(f"      Train={len(train_idx)}, Test={len(test_idx)}")

    x_train_raw = np.ascontiguousarray(x[train_idx])
    y_train = y[train_idx].copy()
    x_test_raw = np.ascontiguousarray(x[test_idx])
    y_test = y[test_idx].copy()
    del x

    print("[4/5] Fitting StandardScaler on training set …")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_test_scaled = scaler.transform(x_test_raw).astype(np.float32)
    joblib.dump(scaler, args.output_dir / "scaler.joblib")
    del x_train_raw, x_test_raw

    # ── Common kwargs shared by all three methods ────────────────────────────
    svm_kwargs = dict(
        use_cv=args.use_cv,
        fixed_c=args.fixed_c,
        c_grid=c_grid,
        cv_folds=args.cv_folds,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    print("[5/5] Running dimensionality-reduction experiments …")
    all_results: List[Dict[str, Any]] = []

    # ── Method 1: SelectKBest ────────────────────────────────────────────────
    if not args.no_selection:
        all_results.extend(
            run_feature_selection(
                x_train_scaled,
                y_train,
                x_test_scaled,
                y_test,
                dims=sel_dims,
                input_dim=input_dim,
                output_dir=args.output_dir,
                **svm_kwargs,
            )
        )
    else:
        print("\n  [SKIP] Feature selection (--no-selection).")

    # ── Method 2: PCA ────────────────────────────────────────────────────────
    if not args.no_pca:
        all_results.extend(
            run_pca(
                x_train_scaled,
                y_train,
                x_test_scaled,
                y_test,
                dims=pca_dims,
                input_dim=input_dim,
                output_dir=args.output_dir,
                **svm_kwargs,
            )
        )
    else:
        print("\n  [SKIP] PCA (--no-pca).")

    # ── Method 3: Sparse coding ──────────────────────────────────────────────
    if not args.no_sparse_coding:
        all_results.extend(
            run_sparse_coding(
                x_train_scaled,
                y_train,
                x_test_scaled,
                y_test,
                dims=sc_dims,
                output_dir=args.output_dir,
                dict_init=args.sc_dict_init,
                omp_k=args.sc_omp_k,
                transform_n_jobs=args.sc_transform_n_jobs,
                save_artifacts=args.sc_save_artifacts,
                **svm_kwargs,
            )
        )
    else:
        print("\n  [SKIP] Sparse coding (--no-sparse-coding).")

    out_path = args.output_dir / "metrics_q3_q4.json"
    if args.merge_existing:
        all_results = merge_with_existing_results(out_path, all_results)

    # ── Q4: find optimal ─────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Q4 -- Optimal Method & Dimensionality")
    print("=" * 62)

    optimal = find_optimal(all_results)
    summary = summarise_by_method(all_results)

    print("\n  Best overall:")
    print(f"     Method        : {optimal['method']}")
    print(f"     Dimension     : {optimal['dim']}")
    print(f"     Test accuracy : {optimal['test_accuracy']:.6f}")
    print(f"     Best C        : {optimal['best_C']}")

    print("\n  Per-method best:")
    for method, info in summary.items():
        cv_str = (
            f"cv_acc={info['best_cv_accuracy']:.6f}"
            if info["best_cv_accuracy"] is not None
            else "cv=N/A (fixed C)"
        )
        print(
            f"     {method:<12s} dim={info['best_dim']:>5d} "
            f"test_acc={info['best_test_accuracy']:.6f}  {cv_str}"
        )

    # ── Save results ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_global
    output = {
        "experiment_settings": {
            "input_dim": input_dim,
            "n_samples": n_samples,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "seed": args.seed,
            "use_cv": args.use_cv,
            "fixed_c": args.fixed_c,
            "c_grid": c_grid,
            "cv_folds": args.cv_folds,
            "max_iter": args.max_iter,
            "sc_dict_init": args.sc_dict_init,
            "sc_omp_k": args.sc_omp_k,
            "sc_transform_n_jobs": args.sc_transform_n_jobs,
            "sel_dims": sel_dims,
            "pca_dims": pca_dims,
            "sc_dims": sc_dims,
        },
        "q4_optimal": {
            "method": optimal["method"],
            "dim": optimal["dim"],
            "test_accuracy": optimal["test_accuracy"],
            "cv_accuracy": optimal["cv_accuracy"],
            "best_C": optimal["best_C"],
        },
        "per_method_summary": summary,
        "all_results": all_results,
        "total_elapsed_seconds": total_elapsed,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Results  → {out_path}")
    print(f"  Elapsed  : {total_elapsed / 60:.1f} min")
    print("  Done.")


if __name__ == "__main__":
    main()
