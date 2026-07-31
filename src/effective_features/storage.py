"""
storage.py - история расчётов в базе данных.

Законченные расчёты складываются в SQLite.

Почему SQLite: она встроена в Python, ставить ничего не надо, а вся база -
один файл рядом с проектом. Для инструмента, которым пользуется один человек
или небольшая группа, этого более чем достаточно.

Про потоки: каждая функция открывает своё соединение и закрывает его.
Так делают потому, что соединение SQLite нельзя передавать между потоками,
а расчёты у нас как раз идут в отдельных потоках.
"""
import os
import json
import sqlite3
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'results', 'history.db'
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    task_id        TEXT PRIMARY KEY,
    created_at     TEXT    NOT NULL,
    preset         TEXT    NOT NULL,
    criteria       TEXT    NOT NULL,
    n_pixels       INTEGER,
    n_classes      INTEGER,
    total_time_sec REAL,
    result_json    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    # Позволяет обращаться к колонкам по имени: row['preset'] вместо row[2].
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Создаёт таблицу, если её ещё нет. Вызывать при старте сервера."""
    with _connect() as conn:
        conn.executescript(SCHEMA)


def save_run(task_id: str, created_at: str, preset: str,
             criteria: List[str], result: Dict[str, Any]):
    """Сохраняет законченный расчёт.

    Полный результат кладём одним JSON - структура у него сложная
    (списки признаков, история по шагам), раскладывать её по колонкам
    смысла нет. А вот сводные поля (пресет, объём, время) дублируем
    отдельными колонками: по ним строится список истории, и лишний раз
    разбирать JSON ради двух чисел не хочется.
    """
    dataset = result.get('dataset', {})
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (task_id, created_at, preset, criteria,
                n_pixels, n_classes, total_time_sec, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                created_at,
                preset,
                ','.join(criteria),
                dataset.get('n_pixels'),
                dataset.get('n_classes'),
                result.get('total_time_sec'),
                json.dumps(result, ensure_ascii=False),
            )
        )


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """Список расчётов, свежие сверху. Без полного результата -
    для списка он не нужен, а весит прилично."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT task_id, created_at, preset, criteria,
                      n_pixels, n_classes, total_time_sec, result_json
               FROM runs ORDER BY created_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()

    items = []
    for r in rows:
        # Точность вытаскиваем из JSON - она нужна в списке, чтобы
        # можно было сравнить прогоны, не открывая каждый.
        try:
            res = json.loads(r['result_json'])
            accuracies = {c['id']: c['accuracy'] for c in res.get('criteria', [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            accuracies = {}

        items.append({
            'task_id': r['task_id'],
            'created_at': r['created_at'],
            'preset': r['preset'],
            'criteria': r['criteria'].split(',') if r['criteria'] else [],
            'n_pixels': r['n_pixels'],
            'n_classes': r['n_classes'],
            'total_time_sec': r['total_time_sec'],
            'accuracies': accuracies,
        })
    return items


def get_run(task_id: str) -> Optional[Dict[str, Any]]:
    """Полный результат одного расчёта. None, если такого нет."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT result_json FROM runs WHERE task_id = ?", (task_id,)
        ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row['result_json'])
    except json.JSONDecodeError:
        return None


def delete_run(task_id: str) -> bool:
    """Удаляет запись. Возвращает True, если что-то удалилось."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM runs WHERE task_id = ?", (task_id,))
        return cur.rowcount > 0


def count_runs() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
