"""
dataset.py - описание датасета.

Чтобы инструмент мог считать чужие данные, всё это вынесено в файл dataset.json, который пользователь кладёт рядом со своими снимками.

Минимальный пример такого файла:
    {
      "name": "Мои снимки",
      "images_dir": "images",
      "masks_dir": "masks",
      "band_order": ["B2", "B3", "B4", "B8"],
      "band_roles": { "blue": "B2", "green": "B3", "red": "B4", "nir": "B8" },
      "classes": { "1": "Лес", "2": "Поле", "3": "Вода" }
    }

Снимки и маски сопоставляются по имени файла. Если имена не совпадают
(как в MultiSenGE, где снимок называется ..._S2_..., а маска ..._GR_...), можно задать два списка файлов - построчно, в согласованном порядке.
"""
import os
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = ('.tif', '.tiff')

# Роли каналов, которые нужны для спектральных индексов.
# Если какой-то роли нет - соответствующий индекс просто не считается.
INDEX_REQUIREMENTS = {
    'NDVI': ('nir', 'red'),
    'NDWI': ('green', 'nir'),
    'NDBI': ('swir1', 'nir'),
}

KNOWN_ROLES = ('blue', 'green', 'red', 'nir', 'swir1', 'swir2')


class DatasetError(Exception):
    """Ошибка в описании датасета. Текст рассчитан на пользователя,
    а не на разработчика: он должен подсказывать, что поправить."""


@dataclass
class DatasetSpec:
    name: str
    root: str
    images_dir: str
    masks_dir: str
    band_order: List[str]
    band_roles: Dict[str, str] = field(default_factory=dict)
    classes: Dict[int, str] = field(default_factory=dict)
    image_list: Optional[str] = None
    mask_list: Optional[str] = None
    feature_bands: Optional[List[str]] = None

    @property
    def images_path(self) -> str:
        return os.path.join(self.root, self.images_dir)

    @property
    def masks_path(self) -> str:
        return os.path.join(self.root, self.masks_dir)

    @property
    def n_bands(self) -> int:
        """Сколько каналов в файле."""
        return len(self.band_order)

    def bands_for_features(self) -> List[str]:
        """Какие каналы становятся признаками.

        Не все: в MultiSenGE, например, каналы 20-метрового разрешения
        (B5, B6, B7, B8A) читаются из файла, но в признаки не идут -
        в работу берутся шесть каналов с назначенными ролями.

        Порядок выбора: явный список feature_bands → каналы с ролями →
        все каналы (если роли вообще не заданы).
        """
        if self.feature_bands:
            return list(self.feature_bands)
        if self.band_roles:
            # По порядку band_order, а не по порядку ролей - так предсказуемее
            used = set(self.band_roles.values())
            return [b for b in self.band_order if b in used]
        return list(self.band_order)

    def band_index(self, role: str) -> Optional[int]:
        """Порядковый номер канала в файле по его роли. None, если роль не задана."""
        band = self.band_roles.get(role)
        if band is None:
            return None
        try:
            return self.band_order.index(band)
        except ValueError:
            return None

    def n_features(self, n_windows: int = 4, stats_per_window: int = 8) -> Tuple[int, int]:
        """Сколько признаков получится: (спектральных, текстурных)."""
        spectral = len(self.bands_for_features()) + len(self.available_indices())
        return spectral, n_windows * stats_per_window

    def available_indices(self) -> List[str]:
        """Какие спектральные индексы можно посчитать на этих данных."""
        out = []
        for name, roles in INDEX_REQUIREMENTS.items():
            if all(self.band_index(r) is not None for r in roles):
                out.append(name)
        return out

    def missing_for_indices(self) -> Dict[str, List[str]]:
        """Каких ролей не хватает для каждого непосчитанного индекса -
        чтобы можно было сказать пользователю, что дописать."""
        out = {}
        for name, roles in INDEX_REQUIREMENTS.items():
            missing = [r for r in roles if self.band_index(r) is None]
            if missing:
                out[name] = missing
        return out


def load_spec(path: str) -> DatasetSpec:
    """Читает dataset.json. Путь - либо к самому файлу, либо к папке с ним."""
    if os.path.isdir(path):
        path = os.path.join(path, 'dataset.json')
    elif not os.path.exists(path):
        raise DatasetError(
            f"Путь не существует: {path}\n"
            f"Укажите папку с данными или файл dataset.json."
        )

    if not os.path.isfile(path):
        raise DatasetError(
            f"Не найден файл описания: {path}\n"
            f"Положите dataset.json в папку с данными - образец есть "
            f"в документации."
        )

    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetError(
            f"{os.path.basename(path)}: ошибка в строке {e.lineno} - {e.msg}.\n"
            f"Проверьте запятые и кавычки."
        )

    root = os.path.dirname(os.path.abspath(path))

    for key in ('images_dir', 'masks_dir', 'band_order'):
        if key not in raw:
            raise DatasetError(f"В описании не хватает поля «{key}».")

    band_order = raw['band_order']
    if not isinstance(band_order, list) or not band_order:
        raise DatasetError("Поле «band_order» должно быть непустым списком названий каналов.")
    if len(set(band_order)) != len(band_order):
        dupes = [b for b in set(band_order) if band_order.count(b) > 1]
        raise DatasetError(f"В «band_order» повторяются каналы: {', '.join(dupes)}.")

    band_roles = raw.get('band_roles', {})
    unknown = [r for r in band_roles if r not in KNOWN_ROLES]
    if unknown:
        raise DatasetError(
            f"Неизвестные роли каналов: {', '.join(unknown)}.\n"
            f"Допустимые: {', '.join(KNOWN_ROLES)}."
        )
    for role, band in band_roles.items():
        if band not in band_order:
            raise DatasetError(
                f"Роль «{role}» указывает на канал «{band}», которого нет в «band_order»."
            )

    classes = {}
    for k, v in raw.get('classes', {}).items():
        try:
            classes[int(k)] = str(v)
        except (TypeError, ValueError):
            raise DatasetError(f"Номер класса должен быть целым числом, а не «{k}».")

    feature_bands = raw.get('feature_bands')
    if feature_bands is not None:
        if not isinstance(feature_bands, list) or not feature_bands:
            raise DatasetError("Поле «feature_bands» должно быть непустым списком.")
        unknown_fb = [b for b in feature_bands if b not in band_order]
        if unknown_fb:
            raise DatasetError(
                f"В «feature_bands» есть каналы, которых нет в «band_order»: "
                f"{', '.join(unknown_fb)}."
            )

    lists = raw.get('lists', {})

    spec = DatasetSpec(
        name=raw.get('name', 'Без названия'),
        root=root,
        images_dir=raw['images_dir'],
        masks_dir=raw['masks_dir'],
        band_order=band_order,
        band_roles=band_roles,
        classes=classes,
        image_list=lists.get('images'),
        mask_list=lists.get('masks'),
        feature_bands=feature_bands,
    )

    if not os.path.isdir(spec.images_path):
        raise DatasetError(f"Папка со снимками не найдена: {spec.images_path}")
    if not os.path.isdir(spec.masks_path):
        raise DatasetError(f"Папка с масками не найдена: {spec.masks_path}")

    return spec


def _read_list(root: str, rel_path: str) -> List[str]:
    full = os.path.join(root, rel_path)
    if not os.path.isfile(full):
        raise DatasetError(f"Список файлов не найден: {full}")
    with open(full, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def _key_from_name(name: str) -> str:
    """Ключ для сопоставления снимка и маски по имени.

    Из имени убираются расширение и служебные пометки вроде _S2_ или _GR_,
    которые различают снимок и маску, но не сам участок. Так пара
    31UFP_20200731_S2_2570_7453.tif ↔ 31UFP_GR_2570_7453.tif
    сходится по остатку.
    """
    stem = os.path.splitext(name)[0]
    stem = re.sub(r'_(S2|S1|GR|MASK|LABEL|IMG|IMAGE)_', '_', stem, flags=re.I)
    stem = re.sub(r'^(GR|MASK|LABEL|IMG|IMAGE)_', '', stem, flags=re.I)
    stem = re.sub(r'_(GR|MASK|LABEL|IMG|IMAGE)$', '', stem, flags=re.I)
    return stem.lower()


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Русское склонение: 1 снимок, 2 снимка, 5 снимков."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


def find_pairs(spec: DatasetSpec) -> List[Tuple[str, str]]:
    """Возвращает пары (имя снимка, имя маски).

    Если в описании заданы списки - берёт их построчно. Иначе сопоставляет
    файлы по имени.
    """
    if spec.image_list and spec.mask_list:
        images = _read_list(spec.root, spec.image_list)
        masks = _read_list(spec.root, spec.mask_list)
        if len(images) != len(masks):
            raise DatasetError(
                f"В списках разное число строк: снимков {len(images)}, "
                f"масок {len(masks)}. Списки должны идти в одном порядке."
            )
        return list(zip(images, masks))

    images = sorted(f for f in os.listdir(spec.images_path)
                    if f.lower().endswith(IMAGE_EXTENSIONS))
    masks = sorted(f for f in os.listdir(spec.masks_path)
                   if f.lower().endswith(IMAGE_EXTENSIONS))

    if not images:
        raise DatasetError(
            f"В папке {spec.images_path} нет файлов .tif - проверьте путь "
            f"и расширения."
        )
    if not masks:
        raise DatasetError(f"В папке {spec.masks_path} нет файлов .tif.")

    by_key = {}
    for m in masks:
        by_key.setdefault(_key_from_name(m), m)

    pairs, orphans = [], []
    for img in images:
        mask = by_key.get(_key_from_name(img))
        if mask is None:
            orphans.append(img)
        else:
            pairs.append((img, mask))

    if not pairs:
        example_img = images[0]
        example_mask = masks[0]
        raise DatasetError(
            f"Не удалось сопоставить снимки и маски по именам.\n"
            f"Пример снимка: {example_img}\n"
            f"Пример маски:  {example_mask}\n"
            f"Либо переименуйте файлы так, чтобы имена совпадали, либо "
            f"задайте в описании поле «lists» с двумя списками файлов."
        )

    if orphans:
        shown = ', '.join(orphans[:3])
        more = f" и ещё {len(orphans) - 3}" if len(orphans) > 3 else ""
        word = _plural(len(orphans), 'снимок', 'снимка', 'снимков')
        verb = _plural(len(orphans), 'остался', 'остались', 'осталось')
        print(f"  Без маски {verb} {len(orphans)} {word}: {shown}{more} - пропускаю")

    return pairs


def describe(spec: DatasetSpec, n_pairs: Optional[int] = None) -> dict:
    """Сводка для показа в интерфейсе."""
    missing = spec.missing_for_indices()
    n_spectral, n_textural = spec.n_features()
    return {
        'name': spec.name,
        'n_bands': spec.n_bands,
        'band_order': spec.band_order,
        'band_roles': spec.band_roles,
        'feature_bands': spec.bands_for_features(),
        'n_spectral': n_spectral,
        'n_textural': n_textural,
        'n_features': n_spectral + n_textural,
        'n_classes': len(spec.classes),
        'classes': [{'id': k, 'name': v} for k, v in sorted(spec.classes.items())],
        'indices_available': spec.available_indices(),
        'indices_missing': [
            {'name': name, 'needs': roles} for name, roles in missing.items()
        ],
        'n_pairs': n_pairs,
    }


def check(path: str) -> dict:
    """Полная проверка датасета: описание, пары файлов, число каналов.

    Возвращает сводку и список замечаний. Замечание - это не ошибка:
    работать можно, но стоит знать.
    """
    spec = load_spec(path)
    pairs = find_pairs(spec)
    notes = []

    missing = spec.missing_for_indices()
    for name, roles in missing.items():
        notes.append(
            f"Индекс {name} не считается - не заданы роли каналов: {', '.join(roles)}"
        )

    if not spec.classes:
        notes.append("Названия классов не заданы - в интерфейсе будут номера")

    # Число каналов проверяем по первому снимку. rasterio импортируем здесь,
    # а не наверху: разбор описания должен работать и без него.
    # Ошибку чтения превращаем в замечание - из-за одного битого файла
    # не должна падать вся проверка.
    try:
        import rasterio
    except ImportError:
        notes.append("Число каналов не проверено: не установлен rasterio")
    else:
        first = os.path.join(spec.images_path, pairs[0][0])
        try:
            with rasterio.open(first) as src:
                actual = src.count
        except Exception as e:
            notes.append(
                f"Не удалось прочитать {pairs[0][0]} для проверки каналов: "
                f"{type(e).__name__}"
            )
        else:
            if actual != spec.n_bands:
                raise DatasetError(
                    f"В описании указано {spec.n_bands} каналов, а в файле "
                    f"{pairs[0][0]} их {actual}. Приведите «band_order» "
                    f"в соответствие с данными."
                )

    return {'dataset': describe(spec, len(pairs)), 'notes': notes}
