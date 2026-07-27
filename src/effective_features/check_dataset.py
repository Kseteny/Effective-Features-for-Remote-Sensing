"""
check_dataset.py — проверка датасета перед запуском.

Читает описание, сопоставляет снимки с масками, сверяет число каналов
и говорит, что получится посчитать на этих данных. Полезно запускать
первым делом на чужих данных: ошибку в описании лучше увидеть сразу,
а не через сорок минут расчёта.

Запуск:
    python -m effective_features.check_dataset            # dataset.json в корне
    python -m effective_features.check_dataset путь/к/данным
"""
import os
import sys

from .dataset import check, DatasetError, _plural


def main(path: str = None):
    if path is None:
        # По умолчанию ищем в корне проекта — на уровень выше src/
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.dirname(os.path.dirname(here))

    print(f"\n  Проверяю: {path}")

    try:
        report = check(path)
    except DatasetError as e:
        print(f"\n  Не получилось:\n")
        for line in str(e).split('\n'):
            print(f"    {line}")
        print()
        return 1

    d = report['dataset']
    pairs = d['n_pairs']

    print(f"\n  {d['name']}")
    print(f"  {'—' * 50}")
    print(f"  Пар снимок-маска: {pairs}")
    print(f"  Каналов в файле: {d['n_bands']} — {', '.join(d['band_order'])}")

    fb = d['feature_bands']
    if len(fb) != d['n_bands']:
        skipped = [b for b in d['band_order'] if b not in fb]
        print(f"  В признаки идут: {', '.join(fb)}")
        print(f"  Не используются: {', '.join(skipped)}")

    if d['band_roles']:
        roles = ', '.join(f"{k} → {v}" for k, v in d['band_roles'].items())
        print(f"  Роли: {roles}")

    if d['indices_available']:
        print(f"  Индексы: {', '.join(d['indices_available'])}")

    n_cls = d['n_classes']
    if n_cls:
        word = _plural(n_cls, 'класс', 'класса', 'классов')
        print(f"  Классов: {n_cls} {word}")
        for c in d['classes'][:5]:
            print(f"    {c['id']:>3}  {c['name']}")
        if n_cls > 5:
            print(f"    ...  и ещё {n_cls - 5}")

    print(f"\n  Признаков будет: {d['n_features']} "
          f"({d['n_spectral']} спектральных + {d['n_textural']} текстурных)")

    if report['notes']:
        print(f"\n  Обратите внимание:")
        for n in report['notes']:
            print(f"    · {n}")

    print(f"\n  Данные в порядке, можно запускать расчёт.\n")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
