import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from loader import get_file_lists, load_pair
from features import compute_basic_features, compute_indices, compute_glcm_features
from classifier import prepare_data, train_and_eval, predict_full_image

def main():
    print("--- СТАРТ ФИНАЛЬНОЙ ОБРАБОТКИ ---")
    s2_list, gr_list = get_file_lists()
    img, mask = load_pair(s2_list[0], gr_list[0])
    
    print("1. Извлечение признаков...")
    ndvi, _ = compute_indices(img)
    mean_img, var_img = compute_basic_features(img[3], L=7)
    contrast, homogeneity = compute_glcm_features(img[3], L=7)
    
    # ОКНО 1: Визуализация признаков
    print("Отрисовка окна 1 (Признаки)...")
    fig1, ax1 = plt.subplots(2, 2, figsize=(10, 8))
    ax1[0, 0].imshow(ndvi, cmap='RdYlGn'); ax1[0, 0].set_title("NDVI"); ax1[0, 0].axis('off')
    ax1[0, 1].imshow(contrast, cmap='magma'); ax1[0, 1].set_title("Контраст GLCM"); ax1[0, 1].axis('off')
    ax1[1, 0].imshow(homogeneity, cmap='viridis'); ax1[1, 0].set_title("Однородность GLCM"); ax1[1, 0].axis('off')
    ax1[1, 1].imshow(mask, cmap='tab20'); ax1[1, 1].set_title("Эталонная маска"); ax1[1, 1].axis('off')
    plt.tight_layout()
    plt.show() # Скрипт замрет, пока ты не закроешь это окно

    print("2. Обучение классификатора...")
    X, y = prepare_data(ndvi, contrast, homogeneity, mean_img, var_img, mask)
    model = train_and_eval(X, y)
    
    # ОКНО 2: Важность признаков
    print("Отрисовка окна 2 (Важность признаков)...")
    features_names = ['NDVI', 'Contrast', 'Homogeneity', 'Mean', 'Variance']
    importances = model.feature_importances_
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(features_names, importances)
    ax2.set_title("Важность признаков для классификации")
    ax2.set_ylabel("Вес признака")
    plt.show() # Скрипт замрет снова

    print("3. Генерация итоговой карты классификации...")
    raw_map = predict_full_image(model, ndvi, contrast, homogeneity, mean_img, var_img)
    predicted_map = median_filter(raw_map, size=5)
    
    # ОКНО 3: Финальное сравнение
    print("Отрисовка окна 3 (Сравнение)...")
    fig3, ax3 = plt.subplots(1, 2, figsize=(14, 7))
    ax3[0].imshow(mask, cmap='tab20'); ax3[0].set_title("Ground Truth (Эталон)"); ax3[0].axis('off')
    ax3[1].imshow(predicted_map, cmap='tab20'); ax3[1].set_title("Predicted Map (После фильтрации)"); ax3[1].axis('off')
    plt.tight_layout()
    print("--- ВСЁ ГОТОВО! ---")
    plt.show()

if __name__ == "__main__":
    main()