# utils/data_loader.py
import pickle
import pandas as pd

def load_model_and_data():
    with open("saved_models/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    
    with open("saved_models/scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
        
    data = pd.read_csv("data/processed/cleaned_data.csv")
    
    return model, scaler, data

full_feature_order = [
    "koi_score", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_period",
    "koi_time0bk", "koi_impact", "koi_duration", "koi_depth", "koi_prad", "koi_teq", 
    "koi_insol", "koi_model_snr", "koi_count", "koi_num_transits", "koi_tce_plnt_num", 
    "koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass", "koi_kepmag", 
    "koi_disposition"
]