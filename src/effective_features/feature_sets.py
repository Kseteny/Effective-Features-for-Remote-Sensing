"""
feature_sets.py - реестр наборов признаков.

КАК ДОБАВИТЬ СВОЙ НАБОР

Создайте папку user_features рядом с папками src и data, положите
туда файл с любым именем:

    # user_features/gradient.py
    import numpy as np
    from effective_features.feature_sets import register

    def compute(ctx):
        gy, gx = np.gradient(ctx.gray)
        return {'Gradient': np.hypot(gx, gy).astype(np.float32)}

    register(
        id='gradient',
        name='Модуль градиента',
        description='Резкость переходов яркости',
        compute=compute,
    )

Функция получает контекст и возвращает словарь: имя признака → массив
той же формы, что и снимок. Всё остальное программа сделает сама.
"""
import os
import sys
import glob
import importlib.util
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class FeatureContext:
    """Что доступно набору признаков при расчёте.
    image - нормализованный снимок (каналы, высота, ширина)
    gray  - яркость, среднее по каналам, которые идут в признаки
    spec  - описание датасета: порядок каналов, их роли
    window_sizes - размеры скользящих окон из настроек
    """
    image: np.ndarray
    gray: np.ndarray
    spec: object = None
    window_sizes: tuple = (3, 5, 7, 9)

    def band(self, role: str):
        """Канал по роли: 'red', 'nir' и так далее. None, если роли нет."""
        if self.spec is None:
            return None
        i = self.spec.band_index(role)
        return None if i is None else self.image[i]


@dataclass
class FeatureSet:
    id: str
    name: str
    description: str
    compute: Callable          # (ctx) → {имя признака: массив}
    needs_channels: bool = False   # нужен многоканальный снимок
    builtin: bool = True

    def is_available(self, ctx: FeatureContext) -> bool:
        if self.needs_channels and ctx.image.ndim != 3:
            return False
        return True


REGISTRY: Dict[str, FeatureSet] = {}


def register(id: str, name: str, description: str, compute: Callable,
             needs_channels: bool = False, builtin: bool = False):
    """Добавляет набор признаков в реестр."""
    if id in REGISTRY and REGISTRY[id].builtin and not builtin:
        print(f"  Набор «{id}» заменён пользовательским")
    REGISTRY[id] = FeatureSet(
        id=id, name=name, description=description,
        compute=compute, needs_channels=needs_channels, builtin=builtin,
    )
    return REGISTRY[id]


def get(set_id: str) -> Optional[FeatureSet]:
    return REGISTRY.get(set_id)


def all_sets() -> List[FeatureSet]:
    return list(REGISTRY.values())


def unknown(ids: List[str]) -> List[str]:
    return [i for i in ids if i not in REGISTRY]


def describe(fs: FeatureSet) -> dict:
    return {
        'id': fs.id,
        'name': fs.name,
        'description': fs.description,
        'needs_channels': fs.needs_channels,
        'builtin': fs.builtin,
    }


def _spectral(ctx: FeatureContext) -> Dict[str, np.ndarray]:
    """Нормализованные каналы и спектральные индексы."""
    from .features import compute_spectral_features
    return compute_spectral_features(ctx.image, ctx.spec)


def _texture(ctx: FeatureContext) -> Dict[str, np.ndarray]:
    """Текстурные признаки: среднее, дисперсия и направленные корреляции.

    Порядок внутри каждого окна сохранён с самой первой версии программы:
    Mean, Var, Rho_Avg, Rho_Range, затем четыре направления. Менять его
    нельзя - номера признаков попали и в документацию, и в сохранённую
    историю расчётов, где хранятся именно номера, а не имена.
    """
    from .features import get_fast_stats, calc_directional_rho
    out = {}
    for w in ctx.window_sizes:
        mean, var = get_fast_stats(ctx.gray, w)
        out[f'Mean_{w}'] = mean.astype(np.float32)
        out[f'Var_{w}'] = var.astype(np.float32)

        rhos = calc_directional_rho(ctx.gray, mean, var, w)
        out[f'Rho_Avg_{w}'] = (
            (rhos['0'] + rhos['90'] + rhos['45'] + rhos['135']) / 4
        ).astype(np.float32)
        out[f'Rho_Range_{w}'] = (
            np.maximum.reduce(list(rhos.values())) -
            np.minimum.reduce(list(rhos.values()))
        ).astype(np.float32)
        for a in ('0', '90', '45', '135'):
            out[f'Rho_{a}_{w}'] = rhos[a].astype(np.float32)
    return out


register(
    id='spectral',
    name='Спектральные',
    description='Нормализованные каналы снимка и индексы NDVI, NDWI, NDBI. '
                'Какие именно - зависит от описания датасета.',
    compute=_spectral,
    needs_channels=True,
    builtin=True,
)

register(
    id='texture',
    name='Текстурные',
    description='Среднее и дисперсия яркости в скользящем окне, корреляция '
                'соседних пикселей по четырём направлениям, их среднее '
                'и разброс. По восемь признаков на каждый размер окна.',
    compute=_texture,
    builtin=True,
)


_user_loaded = False


def load_user_sets(root: str = None) -> List[str]:
    """Подключает наборы из папки user_features рядом с проектом.

    Каждый файл .py оттуда выполняется один раз при первом обращении
    к признакам. Ошибка в пользовательском файле не должна ронять
    расчёт - о ней сообщается, и работа продолжается без этого набора.
    """
    global _user_loaded
    if _user_loaded:
        return []

    if root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(here))

    folder = os.path.join(root, 'user_features')
    _user_loaded = True

    if not os.path.isdir(folder):
        return []

    loaded = []
    for path in sorted(glob.glob(os.path.join(folder, '*.py'))):
        module_name = os.path.splitext(os.path.basename(path))[0]
        if module_name.startswith('_'):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f'user_features.{module_name}', path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded.append(module_name)
        except Exception as e:
            print(f"  Не удалось подключить {os.path.basename(path)}: "
                  f"{type(e).__name__}: {e}")

    if loaded:
        print(f"  Подключены пользовательские наборы: {', '.join(loaded)}")
    return loaded


def fingerprint(set_ids: List[str], window_sizes) -> str:
    """Короткая подпись состава признаков - для имени кеш-файла.
    Если набор признаков изменился, старый кеш использовать нельзя:
    в нём лежат другие признаки под теми же именами файлов.
    """
    import hashlib
    parts = ','.join(sorted(set_ids)) + '|' + '-'.join(str(w) for w in window_sizes)
    return hashlib.md5(parts.encode()).hexdigest()[:6]
