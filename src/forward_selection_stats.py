import numpy as np
import os
import rasterio
from fast_features import extract_all_features, make_feature_sandwich
import matplotlib.pyplot as plt

def calculate_bhatta_dist(X1, X2):
    """Надёжное расстояние Бхаттачария (работает с 1 и N признаками)."""
    if len(X1) < 5 or len(X2) < 5: 
        return 0.0
    
    # 1. Гарантируем 2D форму (N, D)
    if X1.ndim == 1: X1 = X1.reshape(-1, 1)
    if X2.ndim == 1: X2 = X2.reshape(-1, 1)
    
    m1, m2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
    c1 = np.cov(X1, rowvar=False)
    c2 = np.cov(X2, rowvar=False)
    
    # 2. np.cov для одного столбца возвращает скаляр (0D). Превращаем в матрицу 1x1.
    if c1.ndim == 0: c1 = np.array([[c1]])
    if c2.ndim == 0: c2 = np.array([[c2]])
    
    # 3. Регуляризация
    reg = np.eye(c1.shape[0]) * 1e-6
    c1 += reg
    c2 += reg
    
    cov = (c1 + c2) / 2
    try:
        inv_cov = np.linalg.inv(cov)
        diff = m1 - m2
        term1 = 0.125 * diff @ inv_cov @ diff
        term2 = 0.5 * np.log(np.linalg.det(cov) / np.sqrt(np.linalg.det(c1) * np.linalg.det(c2)))
        return float(term1 + term2)
    except np.linalg.LinAlgError:
        return 0.0

def forward_selection_bhatta(dataset, mask, target_classes=(2, 11), eps=0.001, max_features=10):
    """Жадный отбор признаков по приросту расстояния Бхаттачария."""
    h, w, c = dataset.shape
    mask_flat = mask.flatten()
    feat_matrix = dataset.reshape(-1, c)
    
    X_cls1 = feat_matrix[mask_flat == target_classes[0]]
    X_cls2 = feat_matrix[mask_flat == target_classes[1]]
    
    if len(X_cls1) < 10 or len(X_cls2) < 10:
        print(f"❌ Недостаточно пикселей для классов {target_classes}: {len(X_cls1)} и {len(X_cls2)}")
        return [], []
    
    selected = []
    current_dist = 0.0
    history = []
    
    print(f" Начало отбора для классов {target_classes}...")
    
    for step in range(max_features):
        best_feat = -1
        best_gain = -1
        
        for i in range(c):
            if i in selected: continue
            current_indices = selected + [i]
            dist = calculate_bhatta_dist(X_cls1[:, current_indices], X_cls2[:, current_indices])
            gain = dist - current_dist
            
            if gain > best_gain:
                best_gain = gain
                best_feat = i
                
        if best_gain < eps or best_feat == -1:
            print(f"⏹️ Остановка на шаге {step}. Прирост {best_gain:.5f} < {eps}")
            break
            
        selected.append(best_feat)
        current_dist += best_gain
        history.append(current_dist)
        print(f"  Шаг {step+1}: добавлен признак #{best_feat}, Bhatta = {current_dist:.4f} (+{best_gain:.4f})")
        
    return selected, history

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    s2_dir = os.path.join(data_dir, "s2_pref")
    mask_dir = os.path.join(data_dir, "ground_reference")
    
    print(f" Поиск данных в: {data_dir}")
    
    s2_files = [f for f in os.listdir(s2_dir) if f.endswith(('.tif', '.TIF'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.tif', '.TIF'))]
    
    if not s2_files or not mask_files:
        print("❌ Ошибка: Файлы не найдены!")
        return
    
    img_path = os.path.join(s2_dir, s2_files[0])
    mask_path = os.path.join(mask_dir, mask_files[0])
    print(f" Используемые файлы:\n   - {os.path.basename(img_path)}\n   - {os.path.basename(mask_path)}")
    
    # Загрузка
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        with rasterio.open(img_path) as src:
            img_data = src.read()
            img = np.mean(img_data, axis=0).astype(np.float32) if img_data.ndim == 3 else img_data.astype(np.float32)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min)) * 255
            img = img.astype(np.uint8)
    except Exception as e:
        print(f"❌ Ошибка при чтении файлов: {e}")
        return

    if mask.shape != img.shape:
        from scipy.ndimage import zoom
        mask = zoom(mask, (img.shape[0]/mask.shape[0], img.shape[1]/mask.shape[1]), order=0)

    # Проверка классов в маске
    unique_classes = np.unique(mask)
    print(f" Классы в маске: {unique_classes}")
    if 2 not in unique_classes or 11 not in unique_classes:
        print("⚠️ Классы 2 и 11 не найдены в маске! Отбор прерван.")
        return

    # Генерация признаков
    print("\n⚙️ Генерация признаков (окно 15x15)...")
    feat_dict = extract_all_features(img, window_size=15)
    dataset, names = make_feature_sandwich(feat_dict)
    print(f"📦 Сэндвич: {dataset.shape}, Признаки: {names}")
    
    # Отбор
    selected_indices, history = forward_selection_bhatta(dataset, mask, target_classes=(2, 11))
    
    if selected_indices:
        selected_names = [names[i] for i in selected_indices]
        print(f"\n🏆 ИТОГОВЫЙ НАБОР:")
        for i, name in enumerate(selected_names, 1):
            print(f"   {i}. {name}")
    else:
        print("\n️ Признаки не отобраны.")
    
    # Визуализация
    if history:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(history)+1), history, marker='o', linewidth=2)
        plt.title('Прирост расстояния Бхаттачария (Forward Selection)')
        plt.xlabel('Количество признаков')
        plt.ylabel('Bhattacharyya Distance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(project_root, 'output', 'forward_selection_bhatta.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"\n📊 График сохранён: {out_path}")
        plt.show()

if __name__ == "__main__":
    main()