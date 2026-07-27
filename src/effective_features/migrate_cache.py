"""
migrate_cache.py — переименование кеша признаков под новый ключ.

Зачем: в имя кеш-файла добавился отпечаток набора каналов. Раньше файл
назывался «патч__w3-5-7-9_spec.npz», теперь — «патч__w3-5-7-9_spec_93b8bb.npz».
Без переименования старый кеш просто не подхватится, и все признаки
пересчитаются заново — на полном датасете это сорок минут.

Сами признаки не изменились: формулы те же, каналы те же (проверено
побитовым сравнением). Поэтому переименование безопасно.

Запуск:
    python -m effective_features.migrate_cache           # показать, что будет
    python -m effective_features.migrate_cache --do      # переименовать
"""
import os
import sys
import re

from .config import ExperimentConfig


def main(apply: bool = False):
    cfg = ExperimentConfig()
    cfg.resolve_paths(os.path.abspath(__file__))

    new_key = cfg.cache_key()
    # Старый ключ — тот же, но без отпечатка каналов в конце
    old_key = re.sub(r'_[0-9a-f]{6}$', '', new_key)

    if old_key == new_key:
        print("\n  Ключ не менялся — переименовывать нечего.\n")
        return 0

    cache_dir = cfg.cache_dir
    if not os.path.isdir(cache_dir):
        print(f"\n  Папки кеша нет: {cache_dir}\n")
        return 0

    old_suffix = f"__{old_key}.npz"
    new_suffix = f"__{new_key}.npz"

    files = [f for f in os.listdir(cache_dir) if f.endswith(old_suffix)]

    print(f"\n  Папка кеша: {cache_dir}")
    print(f"  Было:  ...{old_suffix}")
    print(f"  Стало: ...{new_suffix}")
    print(f"  Файлов под переименование: {len(files)}")

    if not files:
        already = len([f for f in os.listdir(cache_dir) if f.endswith(new_suffix)])
        if already:
            print(f"  Уже переименовано ранее: {already} файлов.\n")
        else:
            print(f"  Ничего не нашлось — возможно, кеш пуст.\n")
        return 0

    if not apply:
        for f in files[:3]:
            print(f"    {f}")
        if len(files) > 3:
            print(f"    ... и ещё {len(files) - 3}")
        print(f"\n  Это предпросмотр. Чтобы переименовать, запустите с --do\n")
        return 0

    renamed = skipped = 0
    for f in files:
        src = os.path.join(cache_dir, f)
        dst = os.path.join(cache_dir, f[:-len(old_suffix)] + new_suffix)
        if os.path.exists(dst):
            skipped += 1
            continue
        try:
            os.rename(src, dst)
            renamed += 1
        except OSError as e:
            print(f"    Не вышло переименовать {f}: {e}")

    print(f"\n  Переименовано: {renamed}")
    if skipped:
        print(f"  Пропущено (уже были): {skipped}")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main(apply='--do' in sys.argv))
