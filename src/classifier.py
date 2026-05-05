import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib # для сохранения модели в файл
import os

def prepare_data(ndvi, contrast, homogeneity, mean, var, mask):
    """
    Превращает 2D карты признаков в плоскую таблицу (X) и вектор меток (y).
    """
    # Раскладываем матрицы признаков в одномерные массивы и сцепляем их
    # Порядок признаков: NDVI, Contrast, Homogeneity, Mean, Variance
    X = np.stack([
        ndvi.ravel(), 
        contrast.ravel(), 
        homogeneity.ravel(),
        mean.ravel(),
        var.ravel()
    ], axis=1)
    
    y = mask.ravel()
    
    # Игнорируем пиксели с классом 0 (обычно это фон или отсутствие данных)
    valid_idx = y > 0
    
    return X[valid_idx], y[valid_idx]

def train_and_eval(X, y):
    """
    Обучает Random Forest и выводит отчет о точности.
    """
    # Для быстрой проверки ограничиваем выборку (например, 20 000 точек)
    if len(X) > 20000:
        idx = np.random.choice(len(X), 20000, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    # Разделение на обучающую и проверочную выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.3, random_state=42
    )

    print(f"Обучение на {len(X_train)} пикселях...")
    
    # Инициализация модели (100 деревьев — золотой стандарт)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    # Валидация
    y_pred = clf.predict(X_test)
    print("\n[РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ]")
    print(classification_report(y_test, y_pred))
    
    return clf

def predict_full_image(model, ndvi, contrast, homogeneity, mean, var):
    """
    Применяет обученную модель ко всем пикселям исходного изображения
    для формирования итоговой карты.
    """
    shape = ndvi.shape
    
    # Собираем данные в таблицу для предсказания (включая "нулевые" пиксели)
    X_full = np.stack([
        ndvi.ravel(), 
        contrast.ravel(), 
        homogeneity.ravel(),
        mean.ravel(),
        var.ravel()
    ], axis=1)
    
    print("Генерация итоговой карты...")
    y_pred_full = model.predict(X_full)
    
    # Возвращаем вектору форму исходной картинки
    return y_pred_full.reshape(shape)

def save_model(model, filename="land_cover_model.pkl"):
    """Сохраняет обученную модель на диск"""
    joblib.dump(model, filename)
    print(f"Модель сохранена в {filename}")

def load_model(filename="land_cover_model.pkl"):
    """Загружает модель с диска"""
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None