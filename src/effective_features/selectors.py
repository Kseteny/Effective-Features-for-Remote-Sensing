"""
selectors.py — методы отбора признаков и расстояния между классами.

Реализованы два семейства критериев:
  * filter  (по формулам, без обучения) — расстояния Махаланобиса и Бхаттачарьи
  * wrapper (на основе классификатора)  — kNN с кросс-валидацией

РЕЕСТР КРИТЕРИЕВ (SELECTOR_REGISTRY) позволяет добавлять новые методы отбора,
не меняя пайплайн: достаточно написать функцию и зарегистрировать её.
Это и есть «полигон» для сравнения методов отбора.
"""

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from .config import ExperimentConfig, CLASS_NAMES


# ===========================================================================
# СТАТИСТИКА КЛАССОВ И ПОПАРНЫЕ РАССТОЯНИЯ
# ===========================================================================
def calculate_class_stats(dataset, mask, class_id):
    """Вектор средних + ковариационная матрица для пикселей класса."""
    flat = mask.flatten()
    X    = dataset.reshape(-1, dataset.shape[-1])[flat == class_id]
    if len(X) < 5:
        print(f"   Класс {class_id}: недостаточно пикселей ({len(X)})")
        return None, None
    print(f"   Класс {class_id} ({CLASS_NAMES.get(class_id, '?')}): {len(X)} пкс")
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov += np.eye(cov.shape[0]) * 1e-6
    return np.mean(X, axis=0), cov


def mahalanobis_distance(m1, m2, cov1, cov2):
    """D_M = sqrt((m1−m2)ᵀ · ((Σ1+Σ2)/2)⁻¹ · (m1−m2))"""
    try:
        inv = np.linalg.inv((cov1 + cov2) / 2)
        d   = m1 - m2
        return float(np.sqrt(d @ inv @ d))
    except np.linalg.LinAlgError:
        return np.nan


def bhattacharyya_distance(m1, m2, cov1, cov2):
    """
    D_B = (1/8)(m1−m2)ᵀ Σ⁻¹(m1−m2) + (1/2)·ln(detΣ / √(detΣ1·detΣ2)),
    где Σ = (Σ1+Σ2)/2.
    """
    cov  = (cov1 + cov2) / 2
    diff = m1 - m2
    try:
        term1 = 0.125 * float(diff @ np.linalg.inv(cov) @ diff)
        s1, ld1 = np.linalg.slogdet(cov)
        s2, ld2 = np.linalg.slogdet(cov1)
        s3, ld3 = np.linalg.slogdet(cov2)
        if s1 <= 0 or s2 <= 0 or s3 <= 0:
            return np.nan
        return term1 + 0.5 * (ld1 - 0.5 * (ld2 + ld3))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def compute_all_pairwise_distances(stats, classes):
    """Матрицы попарных D_M и D_B для всех классов."""
    n      = len(classes)
    mm, mb = np.zeros((n, n)), np.zeros((n, n))
    lbl    = [f"C{c}" for c in classes]
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i == j or c1 not in stats or c2 not in stats:
                continue
            m1, v1 = stats[c1]['mean'], stats[c1]['cov']
            m2, v2 = stats[c2]['mean'], stats[c2]['cov']
            mm[i, j] = mahalanobis_distance(m1, m2, v1, v2)
            mb[i, j] = bhattacharyya_distance(m1, m2, v1, v2)
    return (pd.DataFrame(mm, index=lbl, columns=lbl),
            pd.DataFrame(mb, index=lbl, columns=lbl))


# ===========================================================================
# FORWARD SELECTION — БХАТТАЧАРЬЯ (filter)
# ===========================================================================
def _bhatta_samples(X1, X2):
    """Расстояние Бхаттачарьи по двум выборкам признаков."""
    if len(X1) < 5 or len(X2) < 5:
        return 0.0
    if X1.ndim == 1: X1 = X1.reshape(-1, 1)
    if X2.ndim == 1: X2 = X2.reshape(-1, 1)
    m1, m2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
    c1, c2 = np.cov(X1, rowvar=False), np.cov(X2, rowvar=False)
    if c1.ndim == 0: c1 = np.array([[float(c1)]])
    if c2.ndim == 0: c2 = np.array([[float(c2)]])
    reg = np.eye(c1.shape[0]) * 1e-6
    c1 += reg; c2 += reg
    cov = (c1 + c2) / 2
    try:
        term1 = 0.125 * float((m1 - m2) @ np.linalg.inv(cov) @ (m1 - m2))
        s1, ld1 = np.linalg.slogdet(cov)
        s2, ld2 = np.linalg.slogdet(c1)
        s3, ld3 = np.linalg.slogdet(c2)
        if s1 <= 0 or s2 <= 0 or s3 <= 0:
            return 0.0
        return float(term1 + 0.5 * (ld1 - 0.5 * (ld2 + ld3)))
    except np.linalg.LinAlgError:
        return 0.0


def forward_selection_bhatta(dataset, mask, cfg: ExperimentConfig):
    """
    Жадный Forward Selection по критерию расстояния Бхаттачарьи (filter).
    Параметры из cfg: bhatta_pair, eps, max_features.
    Возвращает (selected_indices, history_values).
    """
    target = cfg.bhatta_pair
    c      = dataset.shape[-1]
    flat   = mask.flatten()
    X      = dataset.reshape(-1, c)
    X1, X2 = X[flat == target[0]], X[flat == target[1]]

    if len(X1) < 10 or len(X2) < 10:
        print(f"     Мало пикселей: C{target[0]}={len(X1)}, C{target[1]}={len(X2)}")
        return [], []

    selected, cur, history = [], 0.0, []
    print(f"\n  Forward Selection (Бхаттачарья): классы {target}")
    for step in range(cfg.max_features):
        best_f, best_g = -1, -1.0
        for i in range(c):
            if i in selected:
                continue
            gain = _bhatta_samples(X1[:, selected + [i]], X2[:, selected + [i]]) - cur
            if gain > best_g:
                best_g, best_f = gain, i
        if best_g < cfg.eps or best_f == -1:
            print(f"     Остановка на шаге {step}. Прирост {best_g:.5f} < {cfg.eps}")
            break
        selected.append(best_f)
        cur += best_g
        history.append(cur)
        print(f"     Шаг {step + 1}: признак #{best_f:>2d}, D_B={cur:.4f} (+{best_g:.4f})")
    return selected, history


# ===========================================================================
# FORWARD SELECTION — kNN (wrapper)
# ===========================================================================
def forward_selection_knn(dataset, mask, cfg: ExperimentConfig, target_classes=None):
    """
    Forward Selection по точности kNN (k=cfg.knn_k, cfg.knn_cv-fold CV).
    StandardScaler применяется перед kNN. Выборка ограничивается
    cfg.knn_max_samples (только для скорости кросс-валидации).
    Возвращает (selected_indices, history_values).
    """
    c     = dataset.shape[-1]
    flat  = mask.flatten()
    X_all = dataset.reshape(-1, c)[flat > 0]
    y_all = flat[flat > 0]

    if target_classes is not None:
        sel = np.isin(y_all, target_classes)
        X_all, y_all = X_all[sel], y_all[sel]

    if cfg.knn_max_samples is not None and len(X_all) > cfg.knn_max_samples:
        rng = np.random.default_rng(cfg.random_seed)
        idx = rng.choice(len(X_all), cfg.knn_max_samples, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]

    print(f"\n  Forward Selection (kNN): {len(X_all)} пкс, {len(np.unique(y_all))} классов")
    Xs = StandardScaler().fit_transform(X_all)
    selected, cur, history = [], 0.0, []

    for step in range(cfg.max_features):
        best_f, best_a, best_g = -1, -1.0, -1.0
        for i in range(c):
            if i in selected:
                continue
            acc = cross_val_score(
                KNeighborsClassifier(cfg.knn_k, n_jobs=-1),
                Xs[:, selected + [i]], y_all,
                cv=cfg.knn_cv, scoring='accuracy'
            ).mean()
            gain = acc - cur
            if gain > best_g:
                best_g, best_f, best_a = gain, i, acc
        if best_g < cfg.eps or best_f == -1:
            print(f"     Остановка на шаге {step}. Прирост {best_g:.4f} < {cfg.eps}")
            break
        selected.append(best_f)
        cur = best_a
        history.append(cur)
        print(f"  Шаг {step + 1}: признак #{best_f:>2d}, Acc={cur:.4f} (+{best_g:.4f})")
    return selected, history


# ===========================================================================
# РЕЕСТР КРИТЕРИЕВ — расширяемость без правки пайплайна
# ===========================================================================
# Чтобы добавить новый критерий отбора:
#   1. написать функцию forward_selection_X(dataset, mask, cfg, ...) →
#      возвращает (selected_indices, history)
#   2. добавить её сюда строкой 'имя': {'func': ..., 'kind': 'filter'|'wrapper'}
# Пайплайн сам подхватит новый критерий.
SELECTOR_REGISTRY = {
    'bhattacharyya': {
        'func': forward_selection_bhatta,
        'kind': 'filter',
        'needs_target_classes': False,
        'metric_name': 'D_B',
    },
    'knn': {
        'func': forward_selection_knn,
        'kind': 'wrapper',
        'needs_target_classes': True,
        'metric_name': 'Accuracy',
    },
}
