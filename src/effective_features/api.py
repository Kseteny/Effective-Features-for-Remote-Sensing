"""
api.py - веб-сервис поверх расчётного ядра.

Расчёт идёт минутами, а обычный HTTP-запрос столько не живёт, поэтому
запуск и получение результата разделены:
  POST /api/runs             → ставим задачу в фон, сразу отдаём task_id
  GET  /api/runs/{id}        → браузер опрашивает статус
  GET  /api/runs/{id}/result → забирает результат, когда всё посчиталось

Полное описание запросов - в API.md, автоматическая схема - на /api/docs.

Запуск:
    uvicorn effective_features.api:app --reload
"""
import os
import sys
import time
import uuid
import threading
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import storage, dataset, criteria as crit, feature_sets as fsets
from .config import ExperimentConfig, CLASS_NAMES
from .features import load_all_data, subsample_dataset, rebuild_feature_cube
from .selectors import evaluate_feature_set, Cancelled

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Что сделать при запуске сервера и что - при остановке.
    До yield - запуск, после - остановка."""
    storage.init_db()          # создаём таблицу истории, если её ещё нет
    yield


REGISTRY_IDS = tuple(crit.REGISTRY.keys())

# docs_url переносим с /docs: этот адрес занят страницей интерфейса
app = FastAPI(title="Effective Features API", version="1.2",
              docs_url="/api/docs", redoc_url=None, lifespan=lifespan)

# Разрешаем фронтенду (он крутится на другом порту) обращаться к нам.
# В продакшене список адресов надо сузить до реального домена.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    'fast':    {'name': 'Отладочный', 'description': '10 патчей, быстрая проверка',
                'cfg': lambda: ExperimentConfig(use_thinning=True,
                                                thinning_target_patches=10,
                                                max_pixels_total=120_000)},
    'thinned': {'name': 'Основной', 'description': '~150 патчей, все классы гарантированно',
                'cfg': lambda: ExperimentConfig(use_thinning=True, thinning_target_patches=150)},
}

def build_feature_list(window_sizes=(3, 5, 7, 9), use_spectral=True):
    """Собирает список признаков ровно в том же порядке, в каком их
    нумерует расчётное ядро - иначе индексы в результатах разъедутся."""
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


def _project_root() -> str:
    """Корень проекта: на два уровня выше пакета (src/effective_features/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@app.get("/api/features")
def get_features():
    items = build_feature_list()
    return {
        'total': len(items),
        'spectral': sum(1 for i in items if i['group'] == 'spectral'),
        'textural': sum(1 for i in items if i['group'] == 'textural'),
        'items': items,
    }


@app.get("/api/feature-sets")
def get_feature_sets():
    """Наборы признаков: встроенные и добавленные исследователем."""
    fsets.load_user_sets(_project_root())
    return {'items': [fsets.describe(f) for f in fsets.all_sets()]}


@app.get("/api/criteria")
def get_criteria():
    return {'items': [crit.describe(c) for c in crit.all_criteria()]}


@app.get("/api/classes")
def get_classes():
    # Названия берём из описания датасета; если его нет - встроенные
    try:
        names = dataset.load_spec(_project_root()).classes or CLASS_NAMES
    except dataset.DatasetError:
        names = CLASS_NAMES
    return {'items': [{'id': k, 'name': v} for k, v in sorted(names.items())]}


@app.get("/api/presets")
def get_presets():
    return {'items': [{'id': k, 'name': v['name'], 'description': v['description']}
                      for k, v in PRESETS.items()]}


@app.get("/api/dataset")
def get_dataset():
    """Что за данные сейчас подключены и что на них получится посчитать.

    Описание берётся из dataset.json в корне проекта. Если файла нет
    или в нём ошибка - отдаём текст ошибки, чтобы страница могла
    показать его пользователю, а не молча упасть.
    """
    try:
        report = dataset.check(_project_root())
        return {'ok': True, **report}
    except dataset.DatasetError as e:
        return {'ok': False, 'error': str(e)}


class RunRequest(BaseModel):
    preset: str = Field(..., description="fast | thinned")
    criteria: List[str] = Field(default_factory=lambda: list(crit.DEFAULT_SELECTION))
    max_features: Optional[int] = Field(default=None, ge=1, le=41)
    bhatta_pair: Optional[List[int]] = Field(default=None, min_length=2, max_length=2)
    window_sizes: Optional[List[int]] = None
    use_spectral: Optional[bool] = None


class Task:
    """Одна задача расчёта. Живёт в памяти процесса - для учебного проекта
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
    на странице."""

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
            raise Cancelled()

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
            criterion = crit.get(crit_id)
            if criterion is None:
                continue

            base = 0.15 + 0.8 * i / n_crit
            task.set_stage(f'Отбор: {criterion.name}', base)

            t0 = time.perf_counter()
            selected, history = criterion.select(
                dataset, mask, cfg,
                target_classes=target_classes,
                # Признак отмены проверяется внутри циклов отбора,
                # поэтому остановка происходит почти сразу
                should_stop=lambda: task.cancelled)
            dt = time.perf_counter() - t0

            ev = evaluate_feature_set(dataset, mask, selected, cfg,
                                      target_classes=target_classes) or {}
            acc = float(ev.get('accuracy', 0))

            quality_curve = []
            for k in range(1, len(selected) + 1):
                check_cancelled()
                ev_k = evaluate_feature_set(dataset, mask, selected[:k], cfg,
                                            target_classes=target_classes) or {}
                quality_curve.append({
                    'k': k,
                    'accuracy': round(float(ev_k.get('accuracy', 0)), 4),
                    'f1_macro': round(float(ev_k.get('f1_macro', 0)), 4),
                })

            results.append({
                'id': crit_id,
                'name': criterion.name,
                'unit': criterion.unit,
                'color': criterion.color,
                'selected': [int(s) for s in selected],
                'selected_names': [names[s] for s in selected],
                'history': [float(h) for h in history],
                'quality_curve': quality_curve,
                'accuracy': round(acc, 4),
                'error_rate': round(1 - acc, 4),
                'f1_macro': round(float(ev.get('f1_macro', 0)), 4),
                'time_sec': round(dt, 2),
            })

        # Согласованность считается, только когда критериев ровно два -
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

        # Кладём в историю. Если запись не удалась - расчёт всё равно
        # считается успешным: результат уже посчитан и лежит в памяти,
        # ронять его из-за проблем с базой было бы обидно.
        try:
            storage.save_run(
                task_id=task.id,
                created_at=task.created_at.isoformat(),
                preset=task.req.preset,
                criteria=task.req.criteria,
                result=task.result,
            )
        except Exception as e:
            print(f"  Не удалось сохранить в историю: {e}")

    except Cancelled:
        task.status = 'cancelled'
        task.stage = 'Остановлено'
        task.error = None
    except Exception as e:
        task.status = 'failed'
        task.error = f"{type(e).__name__}: {e}"
    finally:
        proxy.unregister()


@app.post("/api/runs", status_code=202)
def create_run(req: RunRequest):
    if req.preset not in PRESETS:
        raise HTTPException(422, f"Неизвестный пресет: {req.preset}")
    unknown = crit.unknown(req.criteria)
    if unknown:
        known = ', '.join(REGISTRY_IDS)
        raise HTTPException(
            422, f"Неизвестные критерии: {', '.join(unknown)}. Доступны: {known}")

    task_id = uuid.uuid4().hex[:8]
    task = Task(task_id, req)
    with TASKS_LOCK:
        TASKS[task_id] = task

    threading.Thread(target=_run_task, args=(task,), daemon=True).start()
    return {'task_id': task_id, 'status': task.status,
            'created_at': task.created_at.isoformat()}


@app.get("/api/runs")
def list_runs(active_only: bool = True):
    """Список задач в памяти сервера.

    Нужен, чтобы страница могла подхватить уже идущий расчёт - например,
    после обновления страницы или если её открыли в другой вкладке.
    Сам расчёт идёт в фоновом потоке и не зависит от того, смотрит
    на него кто-нибудь или нет.
    """
    with TASKS_LOCK:
        tasks = list(TASKS.values())

    items = [
        {
            'task_id': t.id,
            'status': t.status,
            'stage': t.stage,
            'progress': round(t.progress, 3),
            'elapsed_sec': t.elapsed(),
            'preset': t.req.preset,
            'criteria': list(t.req.criteria),
            'created_at': t.created_at.isoformat(),
        }
        for t in tasks
        if not active_only or t.status in ('queued', 'running')
    ]
    items.sort(key=lambda x: x['created_at'], reverse=True)
    return {'items': items, 'total': len(items)}


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
        saved = storage.get_run(task_id)
        if saved is None:
            raise HTTPException(404, "Задача не найдена")
        return saved

    if task.status == 'failed':
        raise HTTPException(409, f"Задача завершилась с ошибкой: {task.error}")
    if task.status != 'done':
        raise HTTPException(409, f"Результат ещё не готов (статус: {task.status})")
    return task.result


@app.get("/api/history")
def get_history(limit: int = 50):
    """Список прошлых расчётов, свежие сверху."""
    if limit < 1 or limit > 200:
        raise HTTPException(422, "limit должен быть от 1 до 200")
    return {'items': storage.list_runs(limit), 'total': storage.count_runs()}


@app.get("/api/history/{task_id}")
def get_history_item(task_id: str):
    """Полный результат сохранённого расчёта."""
    saved = storage.get_run(task_id)
    if saved is None:
        raise HTTPException(404, "Запись не найдена")
    return saved


@app.delete("/api/history/{task_id}")
def delete_history_item(task_id: str):
    """Удаляет расчёт из истории."""
    if not storage.delete_run(task_id):
        raise HTTPException(404, "Запись не найдена")
    return {'task_id': task_id, 'deleted': True}


@app.post("/api/runs/{task_id}/cancel")
def cancel_run(task_id: str):
    """Останавливает расчёт, но оставляет задачу - чтобы страница
    могла показать, что он был отменён, а не просто исчез."""
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "Задача не найдена")
    if task.status in ('done', 'failed', 'cancelled'):
        raise HTTPException(409, f"Расчёт уже завершён (статус: {task.status})")
    task.cancelled = True
    return {'task_id': task_id, 'cancelling': True}


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


WEB_DIR = os.path.join(_project_root(), 'web', '.output', 'public')

NO_UI_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Интерфейс не собран</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:8vh auto;padding:0 1.5rem;
      line-height:1.6;color:#17211C;background:#ECF1EA}
 code{background:#E4EBE3;padding:.15em .4em;border-radius:4px;
      font-family:ui-monospace,monospace;font-size:.9em}
 pre{background:#fff;border:1px solid #D3DCD3;padding:1rem;border-radius:8px;overflow-x:auto}
 a{color:#14664A}
</style></head><body>
<h1>Интерфейс не собран</h1>
<p>Сам расчётный сервер работает - можно посмотреть
<a href="/docs">список запросов</a>. А вот собранных страниц ещё нет.</p>
<p>Чтобы их собрать, выполните из папки <code>web</code>:</p>
<pre>npm install
npm run generate</pre>
<p>После этого обновите страницу.</p>
</body></html>"""


def _safe_path(rel: str):
    """Путь внутри папки с интерфейсом. None, если запрос пытается
    выбраться наружу - например, через «..» в адресе."""
    candidate = os.path.normpath(os.path.join(WEB_DIR, rel))
    root = os.path.normpath(WEB_DIR)
    if candidate != root and not candidate.startswith(root + os.sep):
        return None
    return candidate


@app.get("/{path:path}", include_in_schema=False)
def serve_frontend(path: str):
    """Отдаёт собранный интерфейс.

    Порядок такой: сначала ищем файл, потом index.html внутри папки,
    и в последнюю очередь - заглушку одностраничного приложения.
    Последнее нужно для страниц, которые собираются в браузере:
    на диске их нет, но открываться по прямой ссылке они должны.
    """
    if not os.path.isdir(WEB_DIR):
        return HTMLResponse(NO_UI_PAGE, status_code=200)

    target = _safe_path(path)
    if target is None:
        raise HTTPException(404, "Не найдено")

    if os.path.isfile(target):
        return FileResponse(target)

    index = os.path.join(target, 'index.html')
    if os.path.isfile(index):
        return FileResponse(index)

    fallback = os.path.join(WEB_DIR, '200.html')
    if os.path.isfile(fallback):
        return FileResponse(fallback)

    raise HTTPException(404, "Страница не найдена")
