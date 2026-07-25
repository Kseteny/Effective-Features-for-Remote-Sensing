"""
api.py — веб-сервис поверх расчётного ядра.

Зачем нужен: расчёты идут минутами, а обычный HTTP-запрос столько не живёт.
Поэтому запуск и получение результата разделены:
  POST /api/runs        → ставим задачу в фон, сразу отдаём task_id
  GET  /api/runs/{id}   → браузер опрашивает статус
  GET  /api/runs/{id}/result → забирает результат, когда всё посчиталось

Запуск:
    pip install fastapi uvicorn
    uvicorn effective_features.api:app --reload

Документация появится сама на http://localhost:8000/docs — FastAPI генерирует
её из типов, ничего писать не надо.
"""
import sys
import time
import uuid
import threading
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ExperimentConfig, CLASS_NAMES
from .features import load_all_data, subsample_dataset, rebuild_feature_cube
from .selectors import (
    forward_selection_bhatta, forward_selection_knn, evaluate_feature_set
)

app = FastAPI(title="Effective Features API", version="1.0")

# Разрешаем фронтенду (он крутится на другом порту) обращаться к нам.
# В продакшене список адресов надо сузить до реального домена.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# СПРАВОЧНИКИ
# ===========================================================================
WINDOW_STATS = [
    ('Mean',      'Среднее значение яркости'),
    ('Var',       'Дисперсия яркости'),
    ('Rho_Avg',   'Средняя направленная корреляция'),
    ('Rho_Range', 'Анизотропность текстуры'),
    ('Rho_0',     'Корреляция по горизонтали (0°)'),
    ('Rho_90',    'Корреляция по вертикали (90°)'),
    ('Rho_45',    'Корреляция по диагонали (45°)'),
    ('Rho_135',   'Корреляция по диагонали (135°)'),
]

SPECTRAL_FEATURES = [
    ('Norm_B2',  'Нормализованный канал B2 (синий)'),
    ('Norm_B3',  'Нормализованный канал B3 (зелёный)'),
    ('Norm_B4',  'Нормализованный канал B4 (красный)'),
    ('Norm_B8',  'Нормализованный канал B8 (ближний ИК)'),
    ('Norm_B11', 'Нормализованный канал B11 (SWIR1)'),
    ('Norm_B12', 'Нормализованный канал B12 (SWIR2)'),
    ('NDVI',     'Вегетационный индекс'),
    ('NDWI',     'Индекс водных объектов'),
    ('NDBI',     'Индекс застроенности'),
]

PRESETS = {
    'fast':     {'name': 'Быстрый',           'description': '10 патчей, для проверки',
                 'cfg': lambda: ExperimentConfig(n_patches=10, max_pixels_total=120_000)},
    'research': {'name': 'Исследовательский', 'description': '50 патчей',
                 'cfg': lambda: ExperimentConfig(n_patches=50)},
    'thinned':  {'name': 'Прореживание',      'description': '~150 патчей, все классы гарантированно',
                 'cfg': lambda: ExperimentConfig(use_thinning=True, thinning_target_patches=150)},
    'full':     {'name': 'Полный',            'description': 'весь датасет, считается долго',
                 'cfg': lambda: ExperimentConfig()},
}

CRITERIA = {
    'bhattacharyya': {
        'name': 'Расстояние Бхаттачарьи', 'type': 'filter', 'speed': 'fast', 'pairwise': True,
        'description': 'Оценивает разделимость пары классов. Не привязан к классификатору, работает за секунды.',
    },
    'knn': {
        'name': 'kNN forward selection', 'type': 'wrapper', 'speed': 'slow', 'pairwise': False,
        'description': 'Оценивает все классы сразу через точность классификатора. Точнее, но заметно медленнее.',
    },
}


def build_feature_list(window_sizes=(3, 5, 7, 9), use_spectral=True):
    """Собирает список признаков ровно в том же порядке, в каком их
    нумерует расчётное ядро — иначе индексы в результатах разъедутся."""
    items, idx = [], 0
    if use_spectral:
        for name, desc in SPECTRAL_FEATURES:
            items.append({'index': idx, 'name': name, 'group': 'spectral',
                          'window': None, 'description': desc})
            idx += 1
    for w in window_sizes:
        for stat, desc in WINDOW_STATS:
            items.append({'index': idx, 'name': f'{stat}_{w}', 'group': 'textural',
                          'window': w, 'description': f'{desc}, окно {w}×{w}'})
            idx += 1
    return items


@app.get("/api/features")
def get_features():
    items = build_feature_list()
    return {
        'total': len(items),
        'spectral': sum(1 for i in items if i['group'] == 'spectral'),
        'textural': sum(1 for i in items if i['group'] == 'textural'),
        'items': items,
    }


@app.get("/api/criteria")
def get_criteria():
    return {'items': [{'id': k, **v} for k, v in CRITERIA.items()]}


@app.get("/api/classes")
def get_classes():
    return {'items': [{'id': k, 'name': v} for k, v in sorted(CLASS_NAMES.items())]}


@app.get("/api/presets")
def get_presets():
    return {'items': [{'id': k, 'name': v['name'], 'description': v['description']}
                      for k, v in PRESETS.items()]}


# ===========================================================================
# ЗАДАЧИ
# ===========================================================================
class RunRequest(BaseModel):
    preset: str = Field(..., description="fast | research | thinned | full")
    criteria: List[str] = Field(default=['bhattacharyya', 'knn'])
    max_features: Optional[int] = Field(default=None, ge=1, le=41)
    bhatta_pair: Optional[List[int]] = Field(default=None, min_length=2, max_length=2)
    window_sizes: Optional[List[int]] = None
    use_spectral: Optional[bool] = None


class Task:
    """Одна задача расчёта. Живёт в памяти процесса — для учебного проекта
    этого достаточно, в проде понадобилась бы очередь вроде Celery."""

    def __init__(self, task_id: str, req: RunRequest):
        self.id = task_id
        self.req = req
        self.status = 'queued'
        self.stage = 'Ожидание'
        self.progress = 0.0
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.created_at = datetime.now()
        self.started_at: Optional[float] = None
        self.log = deque(maxlen=40)     # последние строки вывода
        self.cancelled = False

    def elapsed(self) -> float:
        return round(time.perf_counter() - self.started_at, 1) if self.started_at else 0.0

    def set_stage(self, stage: str, progress: float):
        self.stage = stage
        self.progress = progress


TASKS: Dict[str, Task] = {}
TASKS_LOCK = threading.Lock()


class _ThreadStdout:
    """Перехватывает print() из расчётного кода, чтобы показывать живой лог
    на странице, а не гадать, что там происходит.

    Важно: подменяем sys.stdout один раз на весь процесс и раскидываем строки
    по потокам. Наивный contextlib.redirect_stdout здесь не подходит — он
    меняет вывод глобально, поэтому при двух одновременных расчётах их логи
    перемешались бы, а посторонние сообщения попали бы в чужую задачу.
    """

    def __init__(self, original):
        self._original = original
        self._targets: Dict[int, Task] = {}
        self._buffers: Dict[int, str] = {}

    def register(self, task: Task):
        self._targets[threading.get_ident()] = task

    def unregister(self):
        tid = threading.get_ident()
        self._targets.pop(tid, None)
        self._buffers.pop(tid, None)

    def write(self, s):
        tid = threading.get_ident()
        task = self._targets.get(tid)
        if task is None:                      # обычный вывод сервера
            return self._original.write(s)
        buf = self._buffers.get(tid, '') + s
        while '\n' in buf:
            line, buf = buf.split('\n', 1)
            line = line.strip()
            if line:
                task.log.append(line)
        self._buffers[tid] = buf
        return len(s)

    def flush(self):
        self._original.flush()

    def isatty(self):
        return False


_STDOUT_PROXY: Optional[_ThreadStdout] = None
_STDOUT_LOCK = threading.Lock()


def _get_stdout_proxy() -> _ThreadStdout:
    """Ставит перехватчик при первом запуске задачи и переиспользует дальше."""
    global _STDOUT_PROXY
    with _STDOUT_LOCK:
        if _STDOUT_PROXY is None:
            _STDOUT_PROXY = _ThreadStdout(sys.stdout)
            sys.stdout = _STDOUT_PROXY
    return _STDOUT_PROXY


def _build_config(req: RunRequest) -> ExperimentConfig:
    if req.preset not in PRESETS:
        raise ValueError(f"Неизвестный пресет: {req.preset}")
    cfg = PRESETS[req.preset]['cfg']()
    if req.max_features is not None:
        cfg.max_features = req.max_features
    if req.bhatta_pair is not None:
        cfg.bhatta_pair = tuple(req.bhatta_pair)
    if req.window_sizes is not None:
        cfg.window_sizes = tuple(req.window_sizes)
    if req.use_spectral is not None:
        cfg.use_spectral = req.use_spectral
    return cfg


def _run_task(task: Task):
    """Тело фоновой задачи. Выполняется в отдельном потоке."""
    task.status = 'running'
    task.started_at = time.perf_counter()
    proxy = _get_stdout_proxy()
    proxy.register(task)

    def check_cancelled():
        if task.cancelled:
            raise InterruptedError("Задача отменена")

    try:
        cfg = _build_config(task.req)
        cfg.resolve_paths(__file__)

        task.set_stage('Загрузка данных', 0.05)
        X, y, names = load_all_data(cfg)
        check_cancelled()

        X, y = subsample_dataset(X, y, cfg)
        dataset, mask = rebuild_feature_cube(X, y)
        unique_cls = np.unique(y)
        unique_cls = unique_cls[unique_cls > 0]
        target_classes = [int(c) for c in unique_cls]

        results = []
        n_crit = max(len(task.req.criteria), 1)
        for i, crit_id in enumerate(task.req.criteria):
            check_cancelled()
            base = 0.15 + 0.8 * i / n_crit
            task.set_stage(f'Отбор: {CRITERIA.get(crit_id, {}).get("name", crit_id)}', base)

            t0 = time.perf_counter()
            if crit_id == 'bhattacharyya':
                selected, history = forward_selection_bhatta(dataset, mask, cfg)
            elif crit_id == 'knn':
                selected, history = forward_selection_knn(
                    dataset, mask, cfg, target_classes=target_classes)
            else:
                continue
            dt = time.perf_counter() - t0

            ev = evaluate_feature_set(dataset, mask, selected, cfg,
                                      target_classes=target_classes) or {}
            acc = float(ev.get('accuracy', 0))
            results.append({
                'id': crit_id,
                'selected': [int(s) for s in selected],
                'selected_names': [names[s] for s in selected],
                'history': [float(h) for h in history],
                'accuracy': round(acc, 4),
                'error_rate': round(1 - acc, 4),
                'f1_macro': round(float(ev.get('f1_macro', 0)), 4),
                'time_sec': round(dt, 2),
            })

        # Согласованность считается, только когда критериев ровно два —
        # для одного сравнивать не с чем, для трёх и более нужна другая логика.
        agreement = None
        if len(results) == 2:
            a = set(results[0]['selected_names'])
            b = set(results[1]['selected_names'])
            agreement = {
                'both': sorted(a & b),
                'only_first': sorted(a - b),
                'only_second': sorted(b - a),
            }

        task.result = {
            'task_id': task.id,
            'params': {
                'preset': task.req.preset,
                'max_features': cfg.max_features,
                'bhatta_pair': list(cfg.bhatta_pair),
            },
            'dataset': {
                'n_pixels': int(len(y)),
                'n_classes': len(target_classes),
                'n_features': len(names),
                'feature_names': names,
            },
            'criteria': results,
            'agreement': agreement,
            'total_time_sec': task.elapsed(),
        }
        task.set_stage('Готово', 1.0)
        task.status = 'done'

    except InterruptedError as e:
        task.status = 'failed'
        task.error = str(e)
    except Exception as e:
        task.status = 'failed'
        task.error = f"{type(e).__name__}: {e}"
    finally:
        proxy.unregister()


@app.post("/api/runs", status_code=202)
def create_run(req: RunRequest):
    if req.preset not in PRESETS:
        raise HTTPException(422, f"Неизвестный пресет: {req.preset}")
    unknown = [c for c in req.criteria if c not in CRITERIA]
    if unknown:
        raise HTTPException(422, f"Неизвестные критерии: {', '.join(unknown)}")

    task_id = uuid.uuid4().hex[:8]
    task = Task(task_id, req)
    with TASKS_LOCK:
        TASKS[task_id] = task

    threading.Thread(target=_run_task, args=(task,), daemon=True).start()
    return {'task_id': task_id, 'status': task.status,
            'created_at': task.created_at.isoformat()}


@app.get("/api/runs/{task_id}")
def get_run(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "Задача не найдена")
    return {
        'task_id': task.id,
        'status': task.status,
        'stage': task.stage,
        'progress': round(task.progress, 3),
        'log_tail': list(task.log)[-8:],
        'elapsed_sec': task.elapsed(),
        'error': task.error,
    }


@app.get("/api/runs/{task_id}/result")
def get_result(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "Задача не найдена")
    if task.status == 'failed':
        raise HTTPException(409, f"Задача завершилась с ошибкой: {task.error}")
    if task.status != 'done':
        raise HTTPException(409, f"Результат ещё не готов (статус: {task.status})")
    return task.result


@app.delete("/api/runs/{task_id}")
def delete_run(task_id: str):
    with TASKS_LOCK:
        task = TASKS.pop(task_id, None)
    if task is None:
        raise HTTPException(404, "Задача не найдена")
    task.cancelled = True
    return {'task_id': task_id, 'deleted': True}


@app.get("/api/health")
def health():
    with TASKS_LOCK:
        active = sum(1 for t in TASKS.values() if t.status in ('queued', 'running'))
    return {'status': 'ok', 'tasks_total': len(TASKS), 'tasks_active': active}
