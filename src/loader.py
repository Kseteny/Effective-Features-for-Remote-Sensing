import rasterio
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Твой порядок каналов из README
CHANNELS = {
    'B2': 1, 'B3': 2, 'B4': 3, 'B8': 4, 
    'B5': 5, 'B6': 6, 'B7': 7, 'B8A': 8, 
    'B11': 9, 'B12': 10
}

def load_pair(s2_name, gr_name, data_dir=None):
    """Загружает пару: 10-канальный снимок и маску"""
    if data_dir is None:
        data_dir = os.path.join(PROJECT_DIR, "data")
    s2_path = os.path.join(data_dir, "s2_pref", s2_name.strip())
    gr_path = os.path.join(data_dir, "ground_reference", gr_name.strip())
    
    with rasterio.open(s2_path) as src:
        # Читаем все 10 каналов и переводим в float32 для расчетов
        image = src.read().astype(np.float32)
        
    with rasterio.open(gr_path) as src:
        # Читаем маску (там значения классов 1..14)
        mask = src.read(1).astype(np.uint8)
        
    return image, mask

def get_file_lists(lists_dir=None):
    """Читает списки файлов из текстовых документов"""
    if lists_dir is None:
        lists_dir = os.path.join(PROJECT_DIR, "lists")
    with open(os.path.join(lists_dir, "out_s2_pref.txt"), 'r') as f:
        s2_files = f.readlines()
    with open(os.path.join(lists_dir, "out_gr_pref.txt"), 'r') as f:
        gr_files = f.readlines()
    return s2_files, gr_files
