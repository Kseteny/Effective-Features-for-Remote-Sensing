"""
selectors.py — методы отбора признаков и расстояния между классами.

Реализованы два семейства критериев:
  * filter  (по формулам, без обучения) — расстояния Махаланобиса и Бхаттачарьи
  * wrapper (на основе классификатора)  — kNN с кросс-валидацией

РЕЕСТР КРИТЕРИЕВ (SELECTOR_REGISTRY) позволяет добавлять новые методы отбора,
не меняя пайплайн: достаточно написать функцию и зарегистрировать её.
Это и есть «полигон» для сравнения методов отбора.
"""

import warnings

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score

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
def stratified_subsample(X, y, max_samples, seed, min_per_class=50):
    """
    Стратифицированная подвыборка для kNN.

    В отличие от случайного отбора, гарантирует представительство КАЖДОГО
    класса пропорционально его размеру, но не менее min_per_class точек
    (если класс совсем мал — берутся все его точки).

    Это решает две задачи:
      1) редкие классы не исчезают из выборки → корректная кросс-валидация;
      2) kNN остаётся быстрым (общий размер ограничен max_samples).

    Возвращает (X_sub, y_sub).
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    # Сколько точек выделить каждому классу
    quotas = {}
    for cls, cnt in zip(classes, counts):
        proportional = int(round(max_samples * cnt / total))
        quota = max(min_per_class, proportional)
        quotas[cls] = min(quota, cnt)   # не больше, чем есть

    # Если суммарно вышло больше лимита — ужимаем пропорционально
    # (но сохраняя минимум для редких классов)
    planned = sum(quotas.values())
    if planned > max_samples:
        # классы крупнее min_per_class ужимаем, мелкие не трогаем
        big = {c: q for c, q in quotas.items() if q > min_per_class}
        small_total = sum(q for c, q in quotas.items() if q <= min_per_class)
        budget = max_samples - small_total
        big_sum = sum(big.values())
        if big_sum > 0 and budget > 0:
            for c in big:
                quotas[c] = max(min_per_class, int(big[c] * budget / big_sum))

    # Собираем индексы
    idx_all = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        take = min(quotas[cls], len(cls_idx))
        chosen = rng.choice(cls_idx, size=take, replace=False)
        idx_all.append(chosen)
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


def forward_selection_knn(dataset, mask, cfg: ExperimentConfig, target_classes=None):
    """
    Forward Selection по точности kNN (k=cfg.knn_k, cfg.knn_cv-fold CV).
    StandardScaler применяется перед kNN.

    Выборка ограничивается cfg.knn_max_samples СТРАТИФИЦИРОВАННО — каждый
    класс представлен пропорционально, но не менее min_per_class точек.
    Это сохраняет редкие классы и держит kNN вычислимым.

    Идейно: Бхаттачарья (filter) считается на ВСЕХ данных, а kNN (wrapper)
    вынужденно ограничивается — это и есть предмет сравнения вычислительных
    затрат filter- и wrapper-методов.

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
        before = len(X_all)
        X_all, y_all = stratified_subsample(
            X_all, y_all, cfg.knn_max_samples, cfg.random_seed)
        print(f"  Стратифицированная выборка: {before:,} → {len(X_all):,} пкс")

    print(f"\n  Forward Selection (kNN): {len(X_all)} пкс, {len(np.unique(y_all))} классов")
    Xs = StandardScaler().fit_transform(X_all)

    # Число фолдов CV не может превышать размер наименьшего класса.
    # Если редкие классы малочисленны — автоматически уменьшаем cv,
    # иначе sklearn сыплет предупреждениями и CV некорректна.
    _, class_counts = np.unique(y_all, return_counts=True)
    min_class = int(class_counts.min())
    n_folds = max(2, min(cfg.knn_cv, min_class))
    if n_folds < cfg.knn_cv:
        print(f"  (CV уменьшена до {n_folds} фолдов: наименьший класс = {min_class} пкс)")

    selected, cur, history = [], 0.0, []

    for step in range(cfg.max_features):
        best_f, best_a, best_g = -1, -1.0, -1.0
        for i in range(c):
            if i in selected:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')   # глушим шум от мелких классов
                acc = cross_val_score(
                    KNeighborsClassifier(cfg.knn_k, n_jobs=-1),
                    Xs[:, selected + [i]], y_all,
                    cv=n_folds, scoring='accuracy'
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
# ОЦЕНКА ЭФФЕКТИВНОСТИ ОТОБРАННОГО НАБОРА ПРИЗНАКОВ
# ===========================================================================
def evaluate_feature_set(dataset, mask, feature_indices, cfg: ExperimentConfig,
                         target_classes=None):
    """
    Оценивает качество классификации по отобранному набору признаков
    на ОТДЕЛЬНОЙ контрольной выборке (которую классификатор не видел при
    обучении). Это прямой ответ на вопрос об эффективности признаков и
    вероятности ошибки.

    Схема:
      1. данные делятся на обучающую (70%) и контрольную (30%) части
         со стратификацией (пропорции классов сохраняются);
      2. на обучающей части обучается kNN по выбранным признакам;
      3. на контрольной части измеряются точность и доля ошибок.

    Возвращает словарь:
      n_features  — число признаков в наборе
      accuracy    — доля верных ответов на контрольной выборке (0..1)
      error_rate  — вероятность ошибки = 1 − accuracy
      f1_macro    — F1-мера (усреднённая по классам, учитывает дисбаланс)
      n_train, n_test — размеры выборок
    """
    if not feature_indices:
        return None

    c = dataset.shape[-1]
    flat = mask.flatten()
    X_all = dataset.reshape(-1, c)[flat > 0]
    y_all = flat[flat > 0]

    if target_classes is not None:
        sel = np.isin(y_all, target_classes)
        X_all, y_all = X_all[sel], y_all[sel]

    # Ограничиваем объём для скорости (стратифицированно)
    if cfg.knn_max_samples is not None and len(X_all) > cfg.knn_max_samples:
        X_all, y_all = stratified_subsample(
            X_all, y_all, cfg.knn_max_samples, cfg.random_seed)

    # Оставляем только отобранные признаки
    X_sel = X_all[:, feature_indices]

    # Делим на обучающую и контрольную части (контроль = 30%)
    # Стратификация по классам; редкие классы могут мешать — подстрахуемся
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sel, y_all, test_size=0.3,
            random_state=cfg.random_seed, stratify=y_all)
    except ValueError:
        # если стратификация невозможна (класс из 1 элемента) — без неё
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sel, y_all, test_size=0.3, random_state=cfg.random_seed)

    # Масштабируем по обучающей части, применяем к обеим
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Обучаем kNN и оцениваем на контроле
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = KNeighborsClassifier(cfg.knn_k, n_jobs=-1)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)

    acc = float(accuracy_score(y_te, y_pred))
    f1 = float(f1_score(y_te, y_pred, average='macro'))

    return {
        'n_features': len(feature_indices),
        'accuracy': acc,
        'error_rate': 1.0 - acc,
        'f1_macro': f1,
        'n_train': len(y_tr),
        'n_test': len(y_te),
    }


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
