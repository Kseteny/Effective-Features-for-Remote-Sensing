import cv2
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from fast_features import fast_cascade_mean_std, smart_normalize, make_feature_sandwich

def calculate_ndvi(red_band, nir_band):
    """
    Расчет вегетационного индекса NDVI.
    Ожидает массивы в формате float или автоматически переводит их.
    """
    red = red_band.astype(np.float64)
    nir = nir_band.astype(np.float64)
    denominator = nir + red
    # Защита от деления на ноль
    denominator[denominator == 0] = 1e-10
    ndvi = (nir - red) / denominator
    return ndvi

def plot_correlation_matrix(dataset, names, out_path):
    """
    Строит тепловую карту корреляции между всеми признаками в сэндвиче.
    """
    # Превращаем (H, W, C) в (H*W, C)
    flat_data = dataset.reshape(-1, dataset.shape[-1])
    
    # Чтобы не перегружать память, берем случайную выборку пикселей
    sample_size = min(20000, flat_data.shape[0])
    indices = np.random.choice(flat_data.shape[0], sample_size, replace=False)
    sample_data = flat_data[indices]
    
    # Считаем корреляцию Пирсона
    corr_matrix = np.corrcoef(sample_data, rowvar=False)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=names, yticklabels=names, square=True)
    plt.title('Матрица корреляции признаков (Pearson R)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"--- Матрица корреляции сохранена: {out_path}")

def find_input_image(provided_path: str | None) -> str | None:
    if provided_path and os.path.exists(provided_path):
        return provided_path
    
    # Поиск в папке data проекта
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, 'data')
    if os.path.isdir(data_dir):
        for ext in ('*.png', '*.tif', '*.tiff', '*.jpg'):
            files = glob.glob(os.path.join(data_dir, '**', ext), recursive=True)
            if files:
                return files[0]
    return None

def main():
    parser = argparse.ArgumentParser(description='Лаборатория признаков ДЗЗ')
    parser.add_argument('-i', '--input', help='Путь к основному снимку (Red или Gray)')
    parser.add_argument('--nir', help='Путь к NIR каналу для расчета NDVI', default=None)
    parser.add_argument('-w', '--window', type=int, default=15, help='Размер окна')
    args = parser.parse_args()

    # Настройка путей
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, 'output', 'process_results')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Загрузка данных
    inp_path = find_input_image(args.input)
    if not inp_path:
        print("Ошибка: Входной файл не найден.")
        return

    print(f"Обработка файла: {inp_path}")
    # Пробуем загрузить основной канал
    img = cv2.imread(inp_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        import rasterio
        with rasterio.open(inp_path) as src:
            img = src.read(1)

    # 2. Генерация признаков (Среднее и СКО)
    mean_f64, std_f64 = fast_cascade_mean_std(img, args.window)
    
    # Нормализация для визуализации и классификации
    mean_u8 = smart_normalize(mean_f64, 'mean')
    std_u8 = smart_normalize(std_f64, 'std')

    # Словарь для сборки сэндвича
    features = {
        'Original': img,
        f'Mean_{args.window}': mean_u8,
        f'Std_{args.window}': std_u8
    }

    # 3. Расчет NDVI (если передан NIR канал)
    if args.nir and os.path.exists(args.nir):
        nir_img = cv2.imread(args.nir, cv2.IMREAD_UNCHANGED)
        if nir_img is not None:
            ndvi = calculate_ndvi(img, nir_img)
            # NDVI в диапазоне [-1, 1], переводим в 0-255 для сэндвича
            features['NDVI'] = smart_normalize(ndvi, 'signed')
            plt.imsave(os.path.join(out_dir, 'output_ndvi.png'), ndvi, cmap='RdYlGn')
            print("--- NDVI рассчитан и сохранен.")

    # 4. Сборка сэндвича и корреляция
    dataset, names = make_feature_sandwich(features)
    
    # Строим поле корреляции
    plot_correlation_matrix(dataset, names, os.path.join(out_dir, 'correlation_map.png'))

    # 5. Сохранение визуальных полей
    plt.imsave(os.path.join(out_dir, 'output_mean.png'), mean_u8, cmap='gray')
    plt.imsave(os.path.join(out_dir, 'output_std.png'), std_u8, cmap='magma') # 'magma' лучше подсвечивает текстуру

    print(f"Успех! Пакет {dataset.shape} готов.")
    print(f"Все результаты в папке: {out_dir}")

if __name__ == '__main__':
    main()