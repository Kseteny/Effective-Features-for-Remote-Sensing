import cv2
import numpy as np
import os
from fast_features import fast_cascade_mean_std, smart_normalize, make_feature_sandwich
import argparse
import cv2
import glob
import os
import numpy as np
from fast_features import fast_cascade_mean_std, smart_normalize, make_feature_sandwich


def find_input_image(provided_path: str | None) -> str | None:
    if provided_path:
        # если указан относительный путь — попытаться как есть и относительно проекта
        if os.path.isabs(provided_path) and os.path.exists(provided_path):
            return provided_path
        candidate = os.path.join(os.getcwd(), provided_path)
        if os.path.exists(candidate):
            return candidate
        # попробовать в папке data рядом с репозиторием
        repo_root = os.path.dirname(os.path.dirname(__file__))
        data_candidate = os.path.join(repo_root, 'data', provided_path)
        if os.path.exists(data_candidate):
            return data_candidate
        return None

    # если не указан — найти первый попавшийся файл в data/
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, 'data')
    if os.path.isdir(data_dir):
        for ext in ('*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg'):
            # рекурсивный поиск по подпапкам data/
            files = glob.glob(os.path.join(data_dir, '**', ext), recursive=True)
            if files:
                return files[0]
    return None


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description='Обработка фрагмента ДЗЗ — вычисление признаков')
    parser.add_argument('-i', '--input', help='путь к входному изображению (файл или относительный путь)')
    parser.add_argument('-o', '--output', help='папка для результатов (по умолчанию output/process_results)', default=None)
    parser.add_argument('-w', '--window', type=int, default=15, help='размер окна для признаков')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(__file__))
    default_out = os.path.join(repo_root, 'output', 'process_results')
    out_dir = args.output or default_out
    ensure_dir(out_dir)

    print('Текущая рабочая директория:', os.getcwd())
    inp = find_input_image(args.input)
    if inp is None:
        print('Ошибка: входной файл не найден. Укажите --input или поместите изображение в папку data/')
        return

    print('Использую входной файл:', inp)
    image = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
    if image is None:
        # попытаемся открыть через rasterio (TIFF, мультиспектральные данные)
        try:
            import rasterio
            with rasterio.open(inp) as src:
                band1 = src.read(1)
                image = band1
                print('Загружено через rasterio, размер:', image.shape)
        except Exception as e:
            print('Ошибка: Не удалось загрузить изображение через cv2 и rasterio:', type(e).__name__, e)
            return

    # 2. Расчет признаков
    mean_f64, std_f64 = fast_cascade_mean_std(image, window_size=args.window)

    # 3. Нормализация
    mean_u8 = smart_normalize(mean_f64, feature_type='mean')
    std_u8 = smart_normalize(std_f64, feature_type='std', low_q=5, high_q=95)

    # 4. Сборка сэндвича
    dataset, names = make_feature_sandwich({
        'Original': image,
        f'Mean_{args.window}': mean_u8,
        f'Std_{args.window}': std_u8
    })

    # 5. Сохранение результатов в папку вывода
    mean_out = os.path.join(out_dir, 'output_mean.png')
    std_out = os.path.join(out_dir, 'output_std.png')

    def safe_imwrite(path: str, arr: np.ndarray) -> bool:
        # попытка через cv2
        try:
            ok = cv2.imwrite(path, arr)
            if ok:
                print('Сохранено через cv2:', path)
                return True
            print('cv2.imwrite вернул False для', path)
        except Exception as e:
            print('cv2.imwrite исключение для', path, type(e).__name__, e)

        # попытка через matplotlib
        try:
            import matplotlib.pyplot as plt
            if arr.ndim == 2:
                plt.imsave(path, arr, cmap='gray')
            else:
                plt.imsave(path, arr)
            print('Сохранено через matplotlib:', path)
            return True
        except Exception as e:
            print('matplotlib.imsave не сработал для', path, type(e).__name__, e)

        # как последний вариант — сохранить как .npy
        try:
            np.save(path + '.npy', arr)
            print('Сохранено через numpy.save:', path + '.npy')
            return True
        except Exception as e:
            print('numpy.save не сработал для', path, type(e).__name__, e)

        return False

    # привести к uint8 для безопасного сохранения изображений
    if mean_u8.dtype != np.uint8:
        mean_u8 = np.clip(mean_u8, 0, 255).astype(np.uint8)
    if std_u8.dtype != np.uint8:
        std_u8 = np.clip(std_u8, 0, 255).astype(np.uint8)

    print('mean_u8 dtype/range:', mean_u8.dtype, mean_u8.min(), mean_u8.max())
    print('std_u8 dtype/range:', std_u8.dtype, std_u8.min(), std_u8.max())

    ok1 = safe_imwrite(mean_out, mean_u8)
    ok2 = safe_imwrite(std_out, std_u8)
    if not (ok1 and ok2):
        print('Warning: не все файлы удалось сохранить; проверьте права и кодировку пути.')

    print(f'Обработка завершена. Сформирован пакет: {dataset.shape}')
    print('Результаты сохранены в', out_dir)


if __name__ == '__main__':
    main()