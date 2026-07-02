"""
config.py — конфигурация эксперимента и справочники классов.

EFFECTIVE-FEATURES: инструмент сравнительного анализа методов отбора
признаков в задачах дистанционного зондирования Земли.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt


def _ensure_dir(path, attempts=3):
    """
    Надёжно создаёт папку (включая все родительские).
    На Windows с OneDrive/антивирусом makedirs иногда срабатывает не сразу —
    делаем несколько попыток. При неудаче — понятное сообщение.
    """
    path = os.path.abspath(path)
    last_err = None
    for i in range(attempts):
        try:
            os.makedirs(path, exist_ok=True)
            if os.path.isdir(path):
                return path
        except OSError as e:
            last_err = e
            time.sleep(0.3)
    # Последняя попытка — пробросить понятную ошибку
    raise OSError(
        f"Не удалось создать папку:\n  {path}\n"
        f"Возможные причины: путь синхронизируется OneDrive, нет прав, "
        f"или мешает антивирус. Попробуйте переместить проект в локальную "
        f"папку вне OneDrive (например C:\\Projects\\). Исходная ошибка: {last_err}"
    )


# ===========================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ===========================================================================
@dataclass
class ExperimentConfig:
    """
    Все параметры эксперимента в одном месте. Меняя поля, можно запускать
    разные эксперименты, не трогая код пайплайна.

    Примеры:
        cfg = ExperimentConfig()                       # дефолт: весь датасет
        cfg = ExperimentConfig(n_patches=8)            # быстрый прогон (отладка)
        cfg = ExperimentConfig(window_sizes=(5, 9))    # только два окна
    """
    # --- Данные ---
    n_patches: Optional[int] = None
    # None = использовать ВЕСЬ датасет; число = ограничить N патчами
    max_pixels_total: Optional[int] = None
    # None = без ограничения общей выборки; число = субдискретизация
    random_seed: int = 42

    # --- Признаковое пространство (базовый набор = 41) ---
    window_sizes: Tuple[int, ...] = (3, 5, 7, 9)   # 4 окна → 32 текстурных
    use_spectral: bool = True                       # 9 спектральных

    # --- Forward Selection (общие) ---
    max_features: int = 15      # макс. длина отбираемого набора
    eps: float = 0.001          # порог останова по приросту критерия

    # --- Критерий Бхаттачарьи ---
    bhatta_pair: Tuple[int, int] = (2, 11)   # пара классов (застройка/лес)

    # --- Критерий kNN (wrapper) ---
    knn_k: int = 5                            # число соседей
    knn_cv: int = 5                           # число фолдов CV
    knn_max_samples: Optional[int] = 20_000   # лимит выборки только для kNN

    # --- Критерий Бхаттачарьи (продолжение) ---
    bhatta_max_samples: Optional[int] = 20_000
    # Лимит пикселей НА КАЖДЫЙ класс пары для forward_selection_bhatta и
    # calculate_class_stats. Без него на классах с десятками миллионов
    # пикселей (леса, водно-болотные угодья) np.cov упирается в память —
    # именно это стало причиной падения на полном датасете.

    # --- Прореживание (систематическая выборка патчей) ---
    use_thinning: bool = False
    # True → вместо случайных n_patches патчей берём каждый k-й патч по
    # порядку (равномерно по датасету), плюс гарантированно добираем
    # патчи с редкими классами, которых иначе может не быть в выборке.
    # Предложено В.В. Сергеевым как альтернатива и случайной выборке,
    # и обработке всего датасета целиком.
    thinning_target_patches: int = 150
    # Целевое число патчей при прореживании (реальное число может быть
    # чуть больше — за счёт патчей, добавленных ради редких классов).

    # --- Кеширование признаков патчей ---
    # Признаки патча не меняются между запусками, поэтому их можно
    # посчитать один раз и переиспользовать (особенно полезно при сериях
    # с разными seed, где патчи частично пересекаются, и на полном датасете).
    use_cache: bool = True        # читать готовые признаки из кеша
    save_cache: bool = True       # сохранять посчитанные признаки в кеш
    force_recompute: bool = False # пересчитать заново, игнорируя чтение
    #   force_recompute=True перебивает use_cache: признаки считаются заново
    #   и кеш ОБНОВЛЯЕТСЯ (если save_cache=True). Нужно при смене формул.
    cache_dir: Optional[str] = None   # папка кеша (по умолч. <root>/cache)

    # --- Пути (если None — определяются автоматически) ---
    project_root: Optional[str] = None
    output_dir: Optional[str] = None
    results_dir: Optional[str] = None
    run_tag: Optional[str] = None
    # run_tag — метка запуска. Если задана (например 'seed7'), результаты
    # кладутся в отдельную папку results/<run_tag>/, графики туда же.
    # Нужно для серии запусков с разными seed (модуль compare).

    def resolve_paths(self, base_file: str):
        """
        Достраивает пути относительно структуры проекта.
        base_file — путь к вызывающему модулю (обычно pipeline.py).
        Структура: <root>/src/effective_features/<module>.py

        Если задан run_tag — все результаты (txt + графики) кладутся
        в отдельную папку results/<run_tag>/, чтобы запуски не перезатирались.
        """
        if self.project_root is None:
            self.project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(base_file))))

        # Корневая папка results/ — создаём в первую очередь
        results_root = os.path.join(self.project_root, 'results')

        if self.run_tag:
            # Изолированная папка под этот запуск: results/<run_tag>/
            base = os.path.join(results_root, self.run_tag)
            if self.results_dir is None:
                self.results_dir = base
            if self.output_dir is None:
                self.output_dir = base
        else:
            if self.output_dir is None:
                self.output_dir = os.path.join(self.project_root, 'output')
            if self.results_dir is None:
                self.results_dir = results_root

        # Надёжное создание папок (Windows + OneDrive иногда мешают —
        # делаем несколько попыток и даём понятную ошибку)
        for target in (results_root, self.results_dir, self.output_dir):
            _ensure_dir(target)

        # Папка кеша признаков
        if self.cache_dir is None:
            self.cache_dir = os.path.join(self.project_root, 'cache')
        if self.use_cache or self.save_cache:
            _ensure_dir(self.cache_dir)
        return self

    def cache_key(self):
        """
        Идентификатор конфигурации признаков для имени кеш-файла.
        Кеш валиден только при тех же параметрах признакового пространства
        (окна + спектральность). При их изменении ключ другой — старый кеш
        не подхватится, т.к. признаки были бы иными.
        """
        windows = '-'.join(str(w) for w in self.window_sizes)
        spec = 'spec' if self.use_spectral else 'nospec'
        return f"w{windows}_{spec}"


# ===========================================================================
# СПРАВОЧНИКИ КЛАССОВ MultiSenGE
# ===========================================================================
# ВАЖНО: это НЕ стандартная номенклатура CORINE level-3 по порядку (1.1.1,
# 1.1.2, ... — как ошибочно было раньше), а собственная реклассификация
# авторов MultiSenGE (Wenger et al., 2022, ISPRS Ann. V-3-2022, Table 1,
# https://doi.org/10.5194/isprs-annals-V-3-2022-635-2022):
#   Urban Areas (1)             → классы 1-5
#   Agricultural areas (2)      → классы 6-10
#   Forests/semi-natural (3)    → классы 11-12
#   Wetlands (4)                → класс 13
#   Water Surfaces (5)          → класс 14
CLASS_NAMES = {
    1:  'Плотная застройка',
    2:  'Разреженная застройка',
    3:  'Специализированная застройка',
    4:  'Спец. озеленённые территории',
    5:  'Крупные транспортные сети',
    6:  'Пахотные земли',
    7:  'Виноградники',
    8:  'Сады',
    9:  'Луга/пастбища',
    10: 'Рощи, живые изгороди',
    11: 'Леса',
    12: 'Открытые минеральные пространства',
    13: 'Водно-болотные угодья',
    14: 'Водные поверхности',
}

PALETTE = {
    1: '#E76F51', 2: '#E63946', 3: '#FF9F1C', 4: '#FFBE0B',
    5: '#8338EC', 6: '#3A86FF', 7: '#6D6875', 8: '#F4A261',
    9: '#48CAE4', 10: '#023E8A', 11: '#2A9D8F', 12: '#52B788',
    13: '#74C69D', 14: '#8AC926',
}
DEFAULT_COLORS = plt.cm.tab10.colors


def class_label(cls):
    """Короткая подпись класса для легенд графиков."""
    name = CLASS_NAMES.get(cls, f'Класс {cls}')
    return f"C{cls}: {name[:18]}{'…' if len(name) > 18 else ''}"