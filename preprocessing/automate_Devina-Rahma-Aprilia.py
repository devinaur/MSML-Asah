import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(__file__)

INPUT_PATH = os.path.join(BASE_DIR, "..", "phoneprice_raw.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "phoneprice_preprocessing.csv")

def preprocessing_data(input_path: str, output_path: str):

    # Load dataset
    df = pd.read_csv(input_path)

    # Feature engineering
    df['pixel_total'] = df['px_height'] * df['px_width']

    # Pisahkan fitur dan target
    X = df.drop(columns=['price_range'])
    y = df['price_range']

    # Pisahkan fitur numerik untuk scaling 
    binary_features = [
        'blue', 'dual_sim', 'four_g',
        'three_g', 'touch_screen', 'wifi']

    numeric_features = [
        'battery_power', 'clock_speed', 'fc', 'int_memory',
        'm_dep', 'mobile_wt', 'n_cores', 'pc',
        'px_height', 'px_width', 'ram',
        'sc_h', 'sc_w', 'talk_time', 'pixel_total']

    # Scaling fitur numerik
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Gabungkan kembali fitur dan target
    processed_df = X.copy()
    processed_df['price_range'] = y

    # Simpan dataset hasil preprocessing
    processed_df.to_csv(output_path, index=False)

    return processed_df

if __name__ == "__main__":

    preprocessing_data(INPUT_PATH, OUTPUT_PATH)
    
    print("berhasil")