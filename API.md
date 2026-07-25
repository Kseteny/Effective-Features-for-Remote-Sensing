# API инструмента отбора признаков

Базовый адрес при локальной разработке: `http://localhost:8000`

Все ответы — JSON. Все даты — в формате ISO 8601.

---

## Справочники (нужны, чтобы фронт знал, что показывать)

### `GET /api/features`
Список всех признаков, которые умеет считать программа.

```json
{
  "total": 41,
  "spectral": 9,
  "textural": 32,
  "items": [
    { "index": 0, "name": "Norm_B2", "group": "spectral", "window": null,
      "description": "Нормализованный канал B2 (синий)" },
    { "index": 9, "name": "Mean_3", "group": "textural", "window": 3,
      "description": "Среднее значение яркости, окно 3×3" }
  ]
}
```

### `GET /api/criteria`
Доступные критерии отбора — из них собирается «меню» на странице.

```json
{
  "items": [
    { "id": "bhattacharyya", "name": "Расстояние Бхаттачарьи", "type": "filter",
      "speed": "fast", "pairwise": true,
      "description": "Оценивает разделимость пары классов. Не привязан к классификатору." },
    { "id": "knn", "name": "kNN forward selection", "type": "wrapper",
      "speed": "slow", "pairwise": false,
      "description": "Оценивает все классы сразу через точность классификатора." }
  ]
}
```

### `GET /api/classes`
Список классов датасета (для выбора пары у Бхаттачарьи).

```json
{
  "items": [
    { "id": 1, "name": "Плотная застройка" },
    { "id": 2, "name": "Разреженная застройка" }
  ]
}
```

### `GET /api/presets`
Готовые режимы прогона.

```json
{
  "items": [
    { "id": "fast", "name": "Быстрый", "description": "10 патчей, для проверки" },
    { "id": "research", "name": "Исследовательский", "description": "50 патчей" },
    { "id": "thinned", "name": "Прореживание", "description": "~150 патчей, все классы" },
    { "id": "full", "name": "Полный", "description": "весь датасет, долго" }
  ]
}
```

---

## Запуск расчёта

### `POST /api/runs`
Ставит задачу в очередь и **сразу** возвращает её идентификатор. Расчёт идёт в фоне.

Тело запроса:
```json
{
  "preset": "thinned",
  "criteria": ["bhattacharyya", "knn"],
  "max_features": 15,
  "bhatta_pair": [2, 11],
  "window_sizes": [3, 5, 7, 9],
  "use_spectral": true
}
```
Обязательное поле только `preset`. Остальные — необязательные, при отсутствии
берутся значения по умолчанию из пресета.

Ответ `202 Accepted`:
```json
{ "task_id": "a3f2b1c8", "status": "queued", "created_at": "2026-07-25T14:03:00" }
```

### `GET /api/runs/{task_id}`
Статус задачи. Фронт дёргает этот endpoint раз в 1–2 секунды, пока идёт расчёт.

```json
{
  "task_id": "a3f2b1c8",
  "status": "running",
  "stage": "Отбор признаков",
  "progress": 0.55,
  "log_tail": [
    "Шаг 5: признак # 0, Acc=0.7404 (+0.0130)",
    "Шаг 6: признак # 7, Acc=0.7453 (+0.0050)"
  ],
  "elapsed_sec": 74.2,
  "error": null
}
```

`status` принимает значения: `queued`, `running`, `done`, `failed`.
Когда `status = "failed"`, в поле `error` лежит текст ошибки.

### `GET /api/runs/{task_id}/result`
Результаты. Доступны только когда `status = "done"`, иначе `409 Conflict`.

```json
{
  "task_id": "a3f2b1c8",
  "params": { "preset": "thinned", "n_patches": 150 },
  "dataset": { "n_pixels": 9830400, "n_classes": 14, "n_features": 41 },
  "criteria": [
    {
      "id": "bhattacharyya",
      "selected": [17, 5, 2, 9, 33],
      "selected_names": ["Mean_5", "Norm_B12", "Norm_B4", "Mean_3", "Mean_9"],
      "history": [0.4132, 0.6831, 1.1370, 1.2782, 1.3834],
      "accuracy": 0.752,
      "error_rate": 0.248,
      "f1_macro": 0.364,
      "time_sec": 1.1
    }
  ],
  "agreement": {
    "both": ["Mean_3", "Mean_7", "NDBI"],
    "only_first": ["Mean_5", "Mean_9"],
    "only_second": ["Norm_B2"]
  },
  "distances": {
    "class_ids": [1, 2, 3],
    "bhattacharyya": [[0.0, 0.34, 1.18], [0.34, 0.0, 0.77], [1.18, 0.77, 0.0]],
    "mahalanobis": [[0.0, 0.82, 1.00], [0.82, 0.0, 0.80], [1.00, 0.80, 0.0]]
  },
  "total_time_sec": 139.0
}
```

Числовые данные отдаются как есть — графики фронт рисует сам,
картинки с сервера не передаются.

### `DELETE /api/runs/{task_id}`
Отменяет задачу (если ещё выполняется) и удаляет её из памяти.

---

## Коды ответов

| Код | Когда |
|-----|-------|
| 200 | Всё хорошо |
| 202 | Задача принята в работу |
| 404 | Такой задачи нет |
| 409 | Результат ещё не готов |
| 422 | Ошибка в параметрах запроса |

---

## История расчётов

Законченные расчёты сохраняются в базу (SQLite, файл `results/history.db`)
и доступны после перезапуска сервера.

### `GET /api/history?limit=50`
Список прошлых расчётов, свежие сверху.

```json
{
  "total": 12,
  "items": [
    {
      "task_id": "a3f2b1c8",
      "created_at": "2026-07-25T18:00:00",
      "preset": "thinned",
      "criteria": ["bhattacharyya", "knn"],
      "n_pixels": 9830400,
      "n_classes": 14,
      "total_time_sec": 139.0,
      "accuracies": { "bhattacharyya": 0.752, "knn": 0.755 }
    }
  ]
}
```

### `GET /api/history/{task_id}`
Полный результат сохранённого расчёта — тот же формат,
что у `GET /api/runs/{task_id}/result`.

### `DELETE /api/history/{task_id}`
Удаляет запись из истории.

Примечание: `GET /api/runs/{task_id}/result` теперь тоже заглядывает
в историю, если задачи уже нет в памяти. Так ссылка на результат
продолжает работать после перезапуска сервера.
