"""
migrate_cache.py — переименование кеша признаков под текущий ключ.

Зачем: в имени кеш-файла закодирован состав признаков — окна, каналы,
наборы. Если что-то из этого меняется, ключ становится другим, и старый
кеш перестаёт подхватываться. На полном датасете это сорок минут
пересчёта.

Когда переименование безопасно: только если сами признаки не изменились,
а поменялся лишь способ вычисления ключа. Если вы добавили новый набор
признаков или поменяли каналы — переименовывать НЕЛЬЗЯ: кеш действительно
устарел, его нужно пересчитать.

Скрипт показывает, что собирается сделать, и ничего не трогает
без явного согласия.

Запуск:
    python -m effective_features.migrate_cache          # посмотреть
    python -m effective_features.migrate_cache --do     # переименовать
"""
import os
import re
import sys
from collections import Counter

from .config import ExperimentConfig


def _parse(filename: str):
    """Разбирает имя кеш-файла на «патч» и «ключ конфигурации».
    Формат: {патч}__{ключ}.npz
    """
    if not filename.endswith('.npz') or '__' not in filename:
        return None, None
    base = filename[:-len('.npz')]
    patch, _, key = base.rpartition('__')
    return patch, key


def main(apply: bool = False):
    cfg = ExperimentConfig()
    cfg.resolve_paths(os.path.abspath(__file__))

    new_key = cfg.cache_key()
    cache_dir = cfg.cache_dir

    print(f"\n  Папка кеша:   {cache_dir}")
    print(f"  Текущий ключ: {new_key}")

    if not os.path.isdir(cache_dir):
        print("\n  Папки кеша нет — переименовывать нечего.\n")
        return 0

    files = [f for f in os.listdir(cache_dir) if f.endswith('.npz')]
    if not files:
        print("\n  Кеш пуст.\n")
        return 0

    by_key = Counter()
    for f in files:
        _, key = _parse(f)
        if key:
            by_key[key] += 1

    print("\n  Что лежит в кеше:")
    for key, n in by_key.most_common():
        mark = '   ← текущий' if key == new_key else ''
        print(f"    {key:26s} {n:>5} файлов{mark}")

    # Переименовываем только те, у которых совпадает «база» ключа — окна
    # и режим спектральности. Если отличаются и они, признаки точно другие,
    # и переименование было бы подлогом.
    base_new = re.sub(r'_[0-9a-f]{6}$', '', new_key)
    candidates = []
    for f in files:
        patch, key = _parse(f)
        if key is None or key == new_key:
            continue
        if re.sub(r'_[0-9a-f]{6}$', '', key) == base_new:
            candidates.append((f, f"{patch}__{new_key}.npz"))

    if not candidates:
        already = by_key.get(new_key, 0)
        if already:
            print(f"\n  Всё уже под текущим ключом ({already} файлов).\n")
        else:
            print("\n  Подходящих файлов не нашлось. Возможно, изменились окна")
            print("  или состав признаков — тогда кеш устарел и нужен пересчёт.\n")
        return 0

    print(f"\n  Под переименование: {len(candidates)} файлов")
    for old, new in candidates[:3]:
        print(f"    {old}")
        print(f"      → {new}")
    if len(candidates) > 3:
        print(f"    ... и ещё {len(candidates) - 3}")

    if not apply:
        print("\n  Это предпросмотр.")
        print("  Переименовывайте, только если признаки не менялись — иначе")
        print("  к новому ключу привяжутся старые, уже неверные данные.")
        print("  Если уверены: запустите с --do\n")
        return 0

    renamed = skipped = 0
    for old, new in candidates:
        src = os.path.join(cache_dir, old)
        dst = os.path.join(cache_dir, new)
        if os.path.exists(dst):
            skipped += 1
            continue
        try:
            os.rename(src, dst)
            renamed += 1
        except OSError as e:
            print(f"    Не вышло переименовать {old}: {e}")

    print(f"\n  Переименовано: {renamed}")
    if skipped:
        print(f"  Пропущено (файл с таким именем уже был): {skipped}")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main(apply='--do' in sys.argv))
