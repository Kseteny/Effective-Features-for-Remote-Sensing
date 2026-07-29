"""
criteria.py — реестр критериев отбора признаков.

Зачем: раньше список критериев был размазан по коду — описания в api.py,
выбор через if/elif в двух местах, названия в третьем. Добавить новый
критерий означало не забыть про каждое из них.

Теперь всё в одном месте: описание, функция отбора и то, как показывать
результат. Чтобы добавить критерий, достаточно написать функцию отбора
и дописать одну запись в REGISTRY.

Владислав Викторович просил именно меню: «здесь тоже можно сделать целое
меню: вот по такому критерию — это лучше, вот это лучше». Реестр — его
техническая основа.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .selectors import (
    forward_selection_bhatta,
    forward_selection_maha,
    forward_selection_mi,
    forward_selection_knn,
)


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    type: str          # filter — сам по себе; wrapper — через классификатор
    speed: str         # fast | medium | slow
    scope: str         # pair — одна пара классов; all — все классы
    unit: str          # что означает величина в истории отбора
    description: str
    select: Callable   # (dataset, mask, cfg, target_classes, should_stop)
                       #   → (indices, history)
    color: str         # цвет в интерфейсе

    @property
    def classifier_free(self) -> bool:
        """Не зависит ли критерий от конкретного классификатора.

        Это ровно то свойство, ради которого всё затевалось: результат
        такого критерия не поедет, если сменить модель.
        """
        return self.type == 'filter'


def _bhatta(dataset, mask, cfg, target_classes=None, should_stop=None):
    # Бхаттачарья работает по паре классов из настроек, весь набор ей не нужен
    return forward_selection_bhatta(dataset, mask, cfg, should_stop=should_stop)


def _maha(dataset, mask, cfg, target_classes=None, should_stop=None):
    return forward_selection_maha(dataset, mask, cfg,
                                  target_classes=target_classes,
                                  should_stop=should_stop)


def _mi(dataset, mask, cfg, target_classes=None, should_stop=None):
    return forward_selection_mi(dataset, mask, cfg,
                                target_classes=target_classes,
                                should_stop=should_stop)


def _knn(dataset, mask, cfg, target_classes=None, should_stop=None):
    return forward_selection_knn(dataset, mask, cfg,
                                 target_classes=target_classes,
                                 should_stop=should_stop)


REGISTRY: Dict[str, Criterion] = {
    'bhattacharyya': Criterion(
        id='bhattacharyya',
        name='Расстояние Бхаттачарьи',
        type='filter',
        speed='fast',
        scope='pair',
        unit='D_B',
        description=(
            'Оценивает разделимость пары классов через разницу средних '
            'и различие формы распределений. Не привязан к классификатору, '
            'считается за секунды.'
        ),
        select=_bhatta,
        color='forest',
    ),
    'mahalanobis': Criterion(
        id='mahalanobis',
        name='Расстояние Махаланобиса',
        type='filter',
        speed='fast',
        scope='all',
        unit='D_M',
        description=(
            'Расстояние между средними классов с учётом корреляции признаков. '
            'Усредняется по всем парам классов, поэтому охватывает весь набор. '
            'Устойчивее Бхаттачарьи на редких классах.'
        ),
        select=_maha,
        color='plum',
    ),
    'mutual_info': Criterion(
        id='mutual_info',
        name='Взаимная информация',
        type='filter',
        speed='medium',
        scope='all',
        unit='mRMR',
        description=(
            'Оценивает, насколько знание признака уменьшает неопределённость '
            'в классе. Не предполагает нормального распределения, в отличие '
            'от расстояний. Из оценки вычитается похожесть на уже отобранные '
            'признаки, чтобы в набор не попадали дубликаты.'
        ),
        select=_mi,
        color='water',
    ),
    'knn': Criterion(
        id='knn',
        name='Отбор через kNN',
        type='wrapper',
        speed='slow',
        scope='all',
        unit='точность',
        description=(
            'На каждом шаге обучает классификатор ближайших соседей и смотрит '
            'на его точность. Учитывает все классы, но результат зависит '
            'от выбранного классификатора и считается заметно дольше.'
        ),
        select=_knn,
        color='gold',
    ),
}


DEFAULT_SELECTION: List[str] = ['bhattacharyya', 'mahalanobis']


def get(criterion_id: str) -> Optional[Criterion]:
    return REGISTRY.get(criterion_id)


def all_criteria() -> List[Criterion]:
    return list(REGISTRY.values())


def unknown(ids: List[str]) -> List[str]:
    """Какие из переданных идентификаторов не зарегистрированы."""
    return [i for i in ids if i not in REGISTRY]


def describe(criterion: Criterion) -> dict:
    """Описание для интерфейса — без самой функции отбора."""
    return {
        'id': criterion.id,
        'name': criterion.name,
        'type': criterion.type,
        'speed': criterion.speed,
        'scope': criterion.scope,
        'unit': criterion.unit,
        'description': criterion.description,
        'classifier_free': criterion.classifier_free,
        'color': criterion.color,
    }
