import os
import joblib
import random
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import BertTokenizerFast, BertModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import json
import gc

from sklearn.isotonic import IsotonicRegression
from scipy import stats

# ---------- Global Settings ----------
SEED = 42
N_FOLDS = 5
BATCH_SIZE_BERT = 32
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Feature Definitions ---
ORIGINAL_NUM_FEATS = [
    "video_duration", "video_height", "video_width",
    "user_following_count", "user_follower_count",
    "user_likes_count", "user_video_count", "user_digg_count"
]
VISION_FEAT_SHAPE = (256, 768)
AGGREGATED_VISION_DIM = VISION_FEAT_SHAPE[1]

# 全局音頻特徵維度變量
AUDIO_FEAT_DIM = None

# --- Data Paths (Updated to latest paths) ---
# Training Data Paths
TRAIN_MAIN_DATA_PATH = "/ssd3/chunlin/smp_video_2025/train_data_combined/cleaned_data.json"
TRAIN_CAPTION_V4_PATH = "/ssd3/cheng/SMP2025/caption_shortcaption.json"
TRAIN_CAPTION_V3_PATH = "/ssd3/cheng/SMP2025/caption_category.json"
TRAIN_CAPTION_TARGET_PATH = "/ssd3/cheng/SMP2025/caption_target.json"
TRAIN_VISION_FEAT_DIR_BASE = "/ssd3/hsuan/SMP-Video/features256_vitL/train"
TRAIN_AUDIO_FEAT_DIR_BASE = "/ssd3/hsuan/SMP-Video/audio_feature/train"
PRECOMPUTED_CLUSTER_TRAIN_DATA_PATH = "/ssd3/cheng/SMP2025/cheng_model/clusters_all_types_stats_train_only_300cluster.csv"

# Test Data Paths
TEST_MAIN_DATA_PATH = "/ssd3/chunlin/smp_video_2025/test_data_combined/cleaned_data.json"
TEST_CAPTION_V4_PATH = "/ssd3/cheng/SMP2025/captionQ4_test.json"
TEST_CAPTION_V3_PATH = "/ssd3/cheng/SMP2025/captionQ3_test.json"
TEST_CAPTION_Q6_PATH = "/ssd3/cheng/SMP2025/captionQ6_test.json"
TEST_VISION_FEAT_DIR_BASE = "/ssd3/hsuan/SMP-Video/features256_vitL/test"
TEST_AUDIO_FEAT_DIR_BASE = "/ssd3/hsuan/SMP-Video/audio_feature/test"
PRECOMPUTED_CLUSTER_TEST_DATA_PATH = "/ssd3/cheng/SMP2025/cheng_model/clusters_all_types_stats_train_only_300cluster.csv"

# --- Setup Output Directory and Logging ---
current_time_str = time.strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = f"runs/lightgbm_enhanced_cv_run_{current_time_str}"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(BASE_OUTPUT_DIR, "lightgbm_enhanced_cv_log.txt")

def log_print(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")

log_print(f"LightGBM Enhanced Cross-Validation Run ID: {current_time_str}")
log_print(f"Base Output Directory: {os.path.abspath(BASE_OUTPUT_DIR)}")
log_print(f"Number of Folds: {N_FOLDS}")
log_print(f"Setting random seed: {SEED}")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

script_start_time = time.time()

# ---------- Utility Functions ----------
def read_data_json(json_path, cols=None):
    if not os.path.exists(json_path):
        log_print(f"ERROR: File not found - {json_path}")
        return pd.DataFrame(columns=cols) if cols else pd.DataFrame()
    try:
        return pd.read_json(json_path, lines=True)[cols] if cols else pd.read_json(json_path, lines=True)
    except ValueError:
        try:
            return pd.read_json(json_path)[cols] if cols else pd.read_json(json_path)
        except ValueError as e_inner:
            log_print(f"ERROR: Failed to read JSON file {json_path}. Error: {e_inner}")
            return pd.DataFrame(columns=cols) if cols else pd.DataFrame()

def extract_video_id_from_path_series(path_series):
    def extract_id(path):
        if pd.isna(path) or not isinstance(path, str): return None
        try: return os.path.splitext(os.path.basename(path))[0]
        except Exception: return None
    return path_series.apply(extract_id)

def ensure_user_format_series(uid_series):
    def fmt(uid_val):
        if pd.isna(uid_val) or not isinstance(uid_val, (str, int, float)): return "UNKNOWN_USER"
        uid_str = str(uid_val)
        if uid_str.startswith('USER'): return uid_str
        if uid_str.isdigit(): return f"USER{uid_str}"
        return uid_str
    return uid_series.apply(fmt)

def get_bert_embeddings(texts, tokenizer, model, device, max_len, batch_size, desc="Generating BERT embeddings"):
    all_embeddings = []
    if model is None or tokenizer is None:
        log_print(f"Warning: BERT model or tokenizer not available in get_bert_embeddings for {desc}.")
        emb_size = 768
        if model and hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            emb_size = model.config.hidden_size
        return np.zeros((len(texts), emb_size))
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = list(texts[i:i+batch_size])
        inputs = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    if not all_embeddings:
        emb_size = model.config.hidden_size if model and hasattr(model, 'config') else 768
        return np.array([]).reshape(0, emb_size)
    return np.vstack(all_embeddings)

def load_and_aggregate_vision_feature(path, target_dim):
    if pd.isna(path) or not os.path.exists(path): 
        return np.zeros(target_dim)
    try:
        tensor = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 and tensor.shape[0] > 0:
            if tensor.ndim > 2: tensor = tensor.squeeze(0)
            if tensor.ndim == 2:
                aggregated_feat = tensor.mean(dim=0).numpy()
                if aggregated_feat.shape[0] == target_dim: 
                    return aggregated_feat
                else:
                    final_feat = np.zeros(target_dim)
                    copy_len = min(len(aggregated_feat), target_dim)
                    final_feat[:copy_len] = aggregated_feat[:copy_len]
                    return final_feat
        return np.zeros(target_dim)
    except Exception: 
        return np.zeros(target_dim)

def determine_audio_feature_dimension(audio_dir_base, sample_df=None):
    """確定音頻特徵的維度，與聚類代碼保持一致的邏輯"""
    global AUDIO_FEAT_DIM
    
    if AUDIO_FEAT_DIM is not None:
        return AUDIO_FEAT_DIM
    
    # 方法1：從音頻目錄中尋找第一個有效文件
    if os.path.exists(audio_dir_base):
        for f in os.listdir(audio_dir_base):
            if f.endswith('.pt'):
                try:
                    feat_path = os.path.join(audio_dir_base, f)
                    tensor = torch.load(feat_path, map_location=torch.device('cpu'))
                    if isinstance(tensor, torch.Tensor):
                        if tensor.ndim == 1:
                            AUDIO_FEAT_DIM = tensor.shape[0]
                            log_print(f"Detected 1D audio feature dimension: {AUDIO_FEAT_DIM}")
                            return AUDIO_FEAT_DIM
                        elif tensor.ndim == 2:
                            AUDIO_FEAT_DIM = tensor.shape[1]
                            log_print(f"Detected 2D audio feature dimension: {AUDIO_FEAT_DIM}")
                            return AUDIO_FEAT_DIM
                        elif tensor.ndim > 2:
                            if tensor.shape[0] == 1:
                                tensor = tensor.squeeze(0)
                                if tensor.ndim == 1:
                                    AUDIO_FEAT_DIM = tensor.shape[0]
                                else:
                                    AUDIO_FEAT_DIM = tensor.shape[1]
                            else:
                                AUDIO_FEAT_DIM = tensor.shape[-1]
                            log_print(f"Detected high-dim audio feature dimension: {AUDIO_FEAT_DIM}")
                            return AUDIO_FEAT_DIM
                    elif isinstance(tensor, dict):
                        for key_name in ['audio_features', 'audio_embeddings', 'features', 'embeddings', 'last_hidden_state']:
                            if key_name in tensor and isinstance(tensor[key_name], torch.Tensor):
                                val = tensor[key_name]
                                if val.ndim == 1:
                                    AUDIO_FEAT_DIM = val.shape[0]
                                elif val.ndim >= 2:
                                    AUDIO_FEAT_DIM = val.shape[-1]
                                log_print(f"Detected audio feature dimension from dict[{key_name}]: {AUDIO_FEAT_DIM}")
                                return AUDIO_FEAT_DIM
                except Exception as e:
                    log_print(f"Error examining audio feature file {f}: {e}")
                    continue
    
    # 方法2：如果提供了sample_df，嘗試從中檢測
    if sample_df is not None and 'audio_feat_path_lgbm' in sample_df.columns:
        for path in sample_df['audio_feat_path_lgbm'].dropna():
            if os.path.exists(path):
                try:
                    tensor = torch.load(path, map_location=torch.device('cpu'))
                    if isinstance(tensor, torch.Tensor):
                        if tensor.ndim == 1:
                            AUDIO_FEAT_DIM = tensor.shape[0]
                        elif tensor.ndim >= 2:
                            AUDIO_FEAT_DIM = tensor.shape[-1]
                        log_print(f"Detected audio feature dimension from sample: {AUDIO_FEAT_DIM}")
                        return AUDIO_FEAT_DIM
                except Exception:
                    continue
    
    # 如果都失敗了，使用默認值
    AUDIO_FEAT_DIM = 512
    log_print(f"Could not determine audio feature dimension. Using default: {AUDIO_FEAT_DIM}")
    return AUDIO_FEAT_DIM

def load_and_aggregate_audio_feature(path, target_dim):
    """加載和聚合音頻特徵，與聚類代碼保持一致的處理邏輯"""
    if pd.isna(path) or not os.path.exists(path):
        return np.zeros(target_dim)
    try:
        tensor = torch.load(path, map_location=torch.device('cpu'))
        
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 1:
                if tensor.shape[0] == target_dim:
                    return tensor.numpy()
                else:
                    final_feat = np.zeros(target_dim)
                    copy_len = min(tensor.shape[0], target_dim)
                    final_feat[:copy_len] = tensor[:copy_len].numpy()
                    return final_feat
            
            elif tensor.ndim == 2:
                aggregated_feat = tensor.mean(dim=0).numpy()
                if aggregated_feat.shape[0] == target_dim:
                    return aggregated_feat
                else:
                    final_feat = np.zeros(target_dim)
                    copy_len = min(len(aggregated_feat), target_dim)
                    final_feat[:copy_len] = aggregated_feat[:copy_len]
                    return final_feat
            
            elif tensor.ndim > 2:
                if tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                    if tensor.ndim == 1:
                        if tensor.shape[0] == target_dim:
                            return tensor.numpy()
                        else:
                            final_feat = np.zeros(target_dim)
                            copy_len = min(tensor.shape[0], target_dim)
                            final_feat[:copy_len] = tensor[:copy_len].numpy()
                            return final_feat
                    else:
                        aggregated_feat = tensor.mean(dim=0).numpy()
                        if aggregated_feat.shape[0] == target_dim:
                            return aggregated_feat
                        else:
                            final_feat = np.zeros(target_dim)
                            copy_len = min(len(aggregated_feat), target_dim)
                            final_feat[:copy_len] = aggregated_feat[:copy_len]
                            return final_feat
                else:
                    while tensor.ndim > 1:
                        tensor = tensor.mean(dim=0)
                    return tensor.numpy() if tensor.shape[0] == target_dim else np.zeros(target_dim)
        
        elif isinstance(tensor, dict):
            for key_name in ['audio_features', 'audio_embeddings', 'features', 'embeddings', 'last_hidden_state']:
                if key_name in tensor and isinstance(tensor[key_name], torch.Tensor):
                    tensor_val = tensor[key_name]
                    if tensor_val.ndim == 1:
                        if tensor_val.shape[0] == target_dim:
                            return tensor_val.numpy()
                        else:
                            final_feat = np.zeros(target_dim)
                            copy_len = min(tensor_val.shape[0], target_dim)
                            final_feat[:copy_len] = tensor_val[:copy_len].numpy()
                            return final_feat
                    elif tensor_val.ndim >= 2:
                        aggregated_feat = tensor_val.mean(dim=0).numpy() if tensor_val.ndim == 2 else tensor_val.squeeze(0).mean(dim=0).numpy()
                        if aggregated_feat.shape[0] == target_dim:
                            return aggregated_feat
                        else:
                            final_feat = np.zeros(target_dim)
                            copy_len = min(len(aggregated_feat), target_dim)
                            final_feat[:copy_len] = aggregated_feat[:copy_len]
                            return final_feat
        
        return np.zeros(target_dim)
    except Exception as e:
        log_print(f"Error processing audio feature {path}: {e}")
        return np.zeros(target_dim)

def safe_label_encode(series, encoder, name, fill_unknown_value="Unknown", unknown_value_in_encoder_fit=None):
    if unknown_value_in_encoder_fit is None: 
        unknown_value_in_encoder_fit = fill_unknown_value
    try: 
        default_encoded_value = encoder.transform([unknown_value_in_encoder_fit])[0]
    except ValueError:
        if len(encoder.classes_) > 0:
            default_encoded_value = 0
            log_print(f"Warning: For '{name}', unknown placeholder '{unknown_value_in_encoder_fit}' not in encoder.classes_. Mapped to {default_encoded_value}.")
        else:
            default_encoded_value = 0
            log_print(f"CRITICAL WARNING: Encoder for '{name}' has no classes. Defaulting unseen to 0.")
    
    transformed_list = []
    unseen_count = 0
    for label in series.fillna(fill_unknown_value):
        try: 
            transformed_list.append(encoder.transform([label])[0])
        except ValueError:
            transformed_list.append(default_encoded_value)
            unseen_count += 1
    if unseen_count > 0:
        log_print(f"Info: For '{name}', {unseen_count} unseen labels mapped to default ({default_encoded_value}).")
    return pd.Series(transformed_list, index=series.index, dtype=int)

def cleanup_intermediate_columns(df):
    """清理中間處理列以節省內存"""
    cols_to_drop = [
        'text_pair', 'video_id', 'uid_formatted', 
        'vision_feat_path_lgbm', 'audio_feat_path_lgbm',
        'video_category_filled', 'video_category_temp'
    ]
    return df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

# ----------校正器Class函數-------------

class PopularityCalibrator:
    """專門用於popularity預測校正的類"""
    
    def __init__(self, method='ensemble', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.calibrator = None
        self.fitted = False
        
    def fit(self, y_pred, y_true):
        """基於驗證集預測和真實值擬合校正器"""
        y_pred = np.array(y_pred).reshape(-1, 1)
        y_true = np.array(y_true)
        
        if self.method == 'linear':
            from sklearn.linear_model import LinearRegression
            self.calibrator = LinearRegression()
            self.calibrator.fit(y_pred, y_true)
            
        elif self.method == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge
            degree = self.kwargs.get('degree', 2)
            self.poly_features = PolynomialFeatures(degree=degree)
            y_pred_poly = self.poly_features.fit_transform(y_pred)
            self.calibrator = Ridge(alpha=self.kwargs.get('alpha', 1.0))
            self.calibrator.fit(y_pred_poly, y_true)
            
        elif self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred.flatten(), y_true)
            
        elif self.method == 'quantile':
            self._fit_quantile_mapping(y_pred.flatten(), y_true)
            
        elif self.method == 'ensemble':
            self._fit_ensemble(y_pred.flatten(), y_true)
            
        self.fitted = True
        return self
    
    def _fit_quantile_mapping(self, y_pred, y_true):
        """分位數映射校正"""
        n_quantiles = self.kwargs.get('n_quantiles', 10)
        pred_quantiles = np.linspace(0, 1, n_quantiles + 1)
        pred_bins = np.quantile(y_pred, pred_quantiles)
        
        self.quantile_mapping = []
        for i in range(len(pred_bins) - 1):
            mask = (y_pred >= pred_bins[i]) & (y_pred < pred_bins[i + 1])
            if i == len(pred_bins) - 2:
                mask = (y_pred >= pred_bins[i]) & (y_pred <= pred_bins[i + 1])
            
            if np.sum(mask) > 0:
                avg_true = np.mean(y_true[mask])
                avg_pred = np.mean(y_pred[mask])
                self.quantile_mapping.append((pred_bins[i], pred_bins[i + 1], avg_pred, avg_true))
    
    def _fit_ensemble(self, y_pred, y_true):
        """集成多種校正方法"""
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.preprocessing import PolynomialFeatures
        
        self.ensemble_calibrators = {}
        
        # 線性校正
        linear_cal = LinearRegression()
        linear_cal.fit(y_pred.reshape(-1, 1), y_true)
        self.ensemble_calibrators['linear'] = linear_cal
        
        # 多項式校正
        poly_features = PolynomialFeatures(degree=2)
        y_pred_poly = poly_features.fit_transform(y_pred.reshape(-1, 1))
        poly_cal = Ridge(alpha=1.0)
        poly_cal.fit(y_pred_poly, y_true)
        self.ensemble_calibrators['polynomial'] = (poly_features, poly_cal)
        
        # 等渗回歸
        isotonic_cal = IsotonicRegression(out_of_bounds='clip')
        isotonic_cal.fit(y_pred, y_true)
        self.ensemble_calibrators['isotonic'] = isotonic_cal
    
    def transform(self, y_pred):
        """應用校正到新的預測值"""
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before transform")
            
        y_pred = np.array(y_pred)
        
        if self.method == 'linear':
            return self.calibrator.predict(y_pred.reshape(-1, 1))
            
        elif self.method == 'polynomial':
            y_pred_poly = self.poly_features.transform(y_pred.reshape(-1, 1))
            return self.calibrator.predict(y_pred_poly)
            
        elif self.method == 'isotonic':
            return self.calibrator.predict(y_pred)
            
        elif self.method == 'quantile':
            return self._transform_quantile(y_pred)
            
        elif self.method == 'ensemble':
            return self._transform_ensemble(y_pred)
    
    def _transform_quantile(self, y_pred):
        """應用分位數映射"""
        y_calibrated = y_pred.copy()
        for pred_min, pred_max, avg_pred, avg_true in self.quantile_mapping:
            mask = (y_pred >= pred_min) & (y_pred < pred_max)
            if pred_max == self.quantile_mapping[-1][1]:
                mask = (y_pred >= pred_min) & (y_pred <= pred_max)
            adjustment = avg_true - avg_pred
            y_calibrated[mask] = y_pred[mask] + adjustment
        return y_calibrated
    
    def _transform_ensemble(self, y_pred):
        """應用集成校正"""
        predictions = []
        
        # 線性
        linear_pred = self.ensemble_calibrators['linear'].predict(y_pred.reshape(-1, 1))
        predictions.append(linear_pred)
        
        # 多項式
        poly_features, poly_cal = self.ensemble_calibrators['polynomial']
        y_pred_poly = poly_features.transform(y_pred.reshape(-1, 1))
        poly_pred = poly_cal.predict(y_pred_poly)
        predictions.append(poly_pred)
        
        # 等渗回歸
        isotonic_pred = self.ensemble_calibrators['isotonic'].predict(y_pred)
        predictions.append(isotonic_pred)
        
        return np.mean(predictions, axis=0)


# ---------- Global BERT Model and Tokenizer (Load Once) ----------
log_print("Loading BERT model and tokenizer globally...")
bert_tokenizer_global, bert_model_global = None, None
try:
    bert_tokenizer_global = BertTokenizerFast.from_pretrained("bert-large-uncased")
    bert_model_global = BertModel.from_pretrained("bert-large-uncased").to(DEVICE)
    bert_model_global.eval()
    log_print("BERT model and tokenizer loaded successfully.")
except Exception as e:
    log_print(f"Error loading BERT model/tokenizer: {e}. Text features might be affected.")

# ---------- 1. Initial Data Load and High-Cost Feature Extraction ----------
def initial_data_load_and_feat_extract(
    main_data_path, caption_v4_path, caption_v3_path, caption_target_path,
    vision_feat_dir_base, audio_feat_dir_base, cluster_data_path,
    bert_tokenizer, bert_model, device_bert,
    is_train_data=True
    ):
    run_type = "TRAIN_FULL" if is_train_data else "TEST_FULL"
    log_print(f"Starting initial data load and high-cost feature extraction for {run_type}...")

    df_main = read_data_json(main_data_path)
    df_caption_v4 = read_data_json(caption_v4_path, ["pid", "caption"])
    df_caption_v3 = read_data_json(caption_v3_path, ["pid", "caption"])
    df_caption_target = read_data_json(caption_target_path, ["pid", "caption"])

    if df_main.empty:
        log_print(f"CRITICAL: Main data is empty for {run_type} from {main_data_path}.")
        return None, None, None

    # Merge captions and extract category
    df_caption_v3["video_category_raw"] = df_caption_v3["caption"].str.extract(r"This video is (.*)\.")
    df = pd.merge(df_main, df_caption_v4, on="pid", how="left")
    df = pd.merge(df, df_caption_v3[["pid", "video_category_raw"]], on="pid", how="left")
    df = pd.merge(df, df_caption_target[["pid", "caption"]], on="pid", how="left", suffixes=("", "_target"))
    df.reset_index(drop=True, inplace=True)
    log_print(f"{run_type} - Base data loaded. Records: {len(df)}")

    y_target = df['popularity'].copy() if is_train_data and 'popularity' in df.columns else None
    if is_train_data and y_target is None:
        log_print(f"CRITICAL: 'popularity' column missing in {run_type} data.")
        return None, None, None

    # Prepare numerical features (raw, will be scaled per fold)
    numerical_features_df = df[ORIGINAL_NUM_FEATS].copy()
    for col in ORIGINAL_NUM_FEATS:
        numerical_features_df[col] = pd.to_numeric(numerical_features_df[col], errors="coerce")
    numerical_features_df = numerical_features_df.fillna(0).astype("float32")

    # Raw categorical and cluster labels (will be encoded per fold)
    df["video_category_filled"] = df["video_category_raw"].fillna("Unknown")
    raw_video_cat_series = df["video_category_filled"].copy()

    # Cluster Labels
    cluster_label_cols_to_load = ["pid", "text_cluster_label", "visual_cluster_label", "user_cluster_label", "audio_cluster_label"]
    if os.path.exists(cluster_data_path):
        cluster_df_ext = pd.read_csv(cluster_data_path)
        cols_to_merge_from_ext = [col for col in cluster_label_cols_to_load if col in cluster_df_ext.columns]
        if "pid" in cols_to_merge_from_ext:
            df = pd.merge(df, cluster_df_ext[cols_to_merge_from_ext], on="pid", how="left")
        else: 
            log_print(f"Warning: 'pid' not in cluster data {cluster_data_path} for {run_type}.")
    else: 
        log_print(f"Warning: Cluster data file not found for {run_type}: {cluster_data_path}.")

    cluster_fill_unknown_int = -1
    raw_text_cluster_series = df.get("text_cluster_label", pd.Series(cluster_fill_unknown_int, index=df.index)).fillna(cluster_fill_unknown_int).astype(int)
    raw_visual_cluster_series = df.get("visual_cluster_label", pd.Series(cluster_fill_unknown_int, index=df.index)).fillna(cluster_fill_unknown_int).astype(int)
    raw_user_cluster_series = df.get("user_cluster_label", pd.Series(cluster_fill_unknown_int, index=df.index)).fillna(cluster_fill_unknown_int).astype(int)
    raw_audio_cluster_series = df.get("audio_cluster_label", pd.Series(cluster_fill_unknown_int, index=df.index)).fillna(cluster_fill_unknown_int).astype(int)

    # Text Feature Extraction (BERT Embeddings) - ONCE
    log_print(f"{run_type} - Extracting text features using BERT...")
    df["text_pair"] = df.apply(
        lambda row: f"{row['caption'] or ''} [SEP] {row['post_content'] or ''} [SEP] {row['caption_target'] or ''}",
        axis=1
    )
    text_embeddings_array = get_bert_embeddings(
        df['text_pair'], bert_tokenizer, bert_model, device_bert,
        MAX_LEN, BATCH_SIZE_BERT, desc=f"BERT for {run_type}"
    )
    text_feature_names = [f'text_emb_{j}' for j in range(text_embeddings_array.shape[1])]
    text_features_df = pd.DataFrame(text_embeddings_array, columns=text_feature_names, index=df.index)

    # Visual Feature Extraction - ONCE
    log_print(f"{run_type} - Extracting and aggregating visual features...")
    if 'video_path' in df.columns: 
        df['video_id'] = extract_video_id_from_path_series(df.get('video_path'))
    else: 
        df['video_id'] = None
    df['uid_formatted'] = ensure_user_format_series(df.get('uid'))
    df['vision_feat_path_lgbm'] = df.apply(
        lambda row: f"{vision_feat_dir_base}/{row['uid_formatted']}_{row['video_id']}.pt"
        if pd.notna(row.get('uid_formatted')) and pd.notna(row.get('video_id')) else None, 
        axis=1
    )
    vision_features_list = [load_and_aggregate_vision_feature(path, AGGREGATED_VISION_DIM)
                            for path in tqdm(df['vision_feat_path_lgbm'], desc=f"VisualFeat ({run_type})")]
    vision_features_array = np.array(vision_features_list)
    if vision_features_array.ndim == 1 and AGGREGATED_VISION_DIM == 0: 
        vision_features_array = np.empty((len(df), 0))
    elif vision_features_array.shape[1] != AGGREGATED_VISION_DIM:
        log_print(f"Warning: Visual features array dim mismatch for {run_type}. Using zeros.")
        vision_features_array = np.zeros((len(df), AGGREGATED_VISION_DIM))
    vision_feature_names = [f'vision_emb_{j}' for j in range(AGGREGATED_VISION_DIM)]
    vision_features_df = pd.DataFrame(vision_features_array, columns=vision_feature_names, index=df.index)

    # Audio Feature Extraction - ONCE
    log_print(f"{run_type} - Extracting and aggregating audio features...")
    global AUDIO_FEAT_DIM
    if is_train_data:
        # 訓練時確定音頻特徵維度
        df['audio_feat_path_lgbm'] = df.apply(
            lambda row: f"{audio_feat_dir_base}/{row['uid_formatted']}_{row['video_id']}.pt"
            if pd.notna(row.get('uid_formatted')) and pd.notna(row.get('video_id')) else None,
            axis=1
        )
        AUDIO_FEAT_DIM = determine_audio_feature_dimension(audio_feat_dir_base, df)
    else:
        # 測試時使用已確定的維度
        if AUDIO_FEAT_DIM is None:
            log_print(f"Warning: Audio feature dimension not determined for {run_type}. Using default 512.")
            AUDIO_FEAT_DIM = 512
        df['audio_feat_path_lgbm'] = df.apply(
            lambda row: f"{audio_feat_dir_base}/{row['uid_formatted']}_{row['video_id']}.pt"
            if pd.notna(row.get('uid_formatted')) and pd.notna(row.get('video_id')) else None,
            axis=1
        )
    
    log_print(f"Using audio feature dimension: {AUDIO_FEAT_DIM}")
    
    audio_features_list = [load_and_aggregate_audio_feature(path, AUDIO_FEAT_DIM)
                          for path in tqdm(df['audio_feat_path_lgbm'], desc=f"AudioFeat ({run_type})")]
    audio_features_array = np.array(audio_features_list)
    
    if audio_features_array.ndim == 1:
        log_print(f"Warning: Audio features array is 1D. Creating empty array.")
        audio_features_array = np.empty((len(df), 0))
    elif audio_features_array.shape[1] != AUDIO_FEAT_DIM:
        log_print(f"Warning: Audio features array dim mismatch for {run_type}. Using zeros.")
        audio_features_array = np.zeros((len(df), AUDIO_FEAT_DIM))
    
    audio_feature_names = [f'audio_emb_{j}' for j in range(AUDIO_FEAT_DIM)]
    audio_features_df = pd.DataFrame(audio_features_array, columns=audio_feature_names, index=df.index)

    # Combine all pre-extracted features into one DataFrame for easy slicing in CV
    df_for_cv = pd.concat([
        numerical_features_df,  # Raw numerical
        text_features_df,       # Pre-extracted
        vision_features_df,     # Pre-extracted
        audio_features_df       # Pre-extracted
    ], axis=1)
    
    # Add raw categorical/cluster series to be processed per fold
    df_for_cv['video_category_raw_val'] = raw_video_cat_series
    df_for_cv['text_cluster_raw_val'] = raw_text_cluster_series
    df_for_cv['visual_cluster_raw_val'] = raw_visual_cluster_series
    df_for_cv['user_cluster_raw_val'] = raw_user_cluster_series
    df_for_cv['audio_cluster_raw_val'] = raw_audio_cluster_series

    # 清理中間變量以節省內存
    df = cleanup_intermediate_columns(df)
    
    pids_for_submission = df['pid'].copy() if 'pid' in df else None
    log_print(f"{run_type} - Initial data load and high-cost feature extraction complete. Shape: {df_for_cv.shape}")
    return df_for_cv, y_target, pids_for_submission

# ---------- 1.1 Load and Pre-extract Features for Training Data ----------
log_print("="*60)
log_print("STEP 1: Loading and Pre-extracting Training Data Features")
log_print("="*60)

df_train_full_features, y_train_full, _ = initial_data_load_and_feat_extract(
    main_data_path=TRAIN_MAIN_DATA_PATH,
    caption_v4_path=TRAIN_CAPTION_V4_PATH,
    caption_v3_path=TRAIN_CAPTION_V3_PATH,
    caption_target_path=TRAIN_CAPTION_TARGET_PATH,
    vision_feat_dir_base=TRAIN_VISION_FEAT_DIR_BASE,
    audio_feat_dir_base=TRAIN_AUDIO_FEAT_DIR_BASE,
    cluster_data_path=PRECOMPUTED_CLUSTER_TRAIN_DATA_PATH,
    bert_tokenizer=bert_tokenizer_global, 
    bert_model=bert_model_global, 
    device_bert=DEVICE,
    is_train_data=True
)

if df_train_full_features is None:
    log_print("CRITICAL: Training data full feature extraction failed. Exiting.")
    if bert_model_global is not None: del bert_model_global
    if bert_tokenizer_global is not None: del bert_tokenizer_global
    torch.cuda.empty_cache()
    exit()

log_print(f"Training data processing completed. Final shape: {df_train_full_features.shape}")

# ---------- 1.2 Load and Pre-extract Features for Test Data ----------
log_print("="*60)
log_print("STEP 2: Loading and Pre-extracting Test Data Features")
log_print("="*60)

df_test_full_features, _, pids_test = initial_data_load_and_feat_extract(
    main_data_path=TEST_MAIN_DATA_PATH,
    caption_v4_path=TEST_CAPTION_V4_PATH,
    caption_v3_path=TEST_CAPTION_V3_PATH,
    caption_target_path=TEST_CAPTION_Q6_PATH,
    vision_feat_dir_base=TEST_VISION_FEAT_DIR_BASE,
    audio_feat_dir_base=TEST_AUDIO_FEAT_DIR_BASE,
    cluster_data_path=PRECOMPUTED_CLUSTER_TEST_DATA_PATH,
    bert_tokenizer=bert_tokenizer_global, 
    bert_model=bert_model_global, 
    device_bert=DEVICE,
    is_train_data=False
)

if df_test_full_features is None:
    log_print("CRITICAL: Test data full feature extraction failed. Exiting.")
    if bert_model_global is not None: del bert_model_global
    if bert_tokenizer_global is not None: del bert_tokenizer_global
    torch.cuda.empty_cache()
    exit()

log_print(f"Test data processing completed. Final shape: {df_test_full_features.shape}")

# Free up global BERT model as embeddings are now extracted
if bert_model_global is not None: 
    del bert_model_global
    bert_model_global = None
if bert_tokenizer_global is not None: 
    del bert_tokenizer_global
    bert_tokenizer_global = None
torch.cuda.empty_cache()
gc.collect()
log_print("Global BERT model and tokenizer removed from memory after feature extraction.")

# 保存音頻特徵維度信息
with open(os.path.join(BASE_OUTPUT_DIR, "audio_feature_dim.txt"), 'w') as f:
    f.write(str(AUDIO_FEAT_DIM))
log_print(f"Audio feature dimension saved: {AUDIO_FEAT_DIM}")

# ---------- 2. Per-Fold Preprocessing (Scaling and Encoding) ----------
def preprocess_data_for_fold(df_train_fold_feats, df_val_fold_feats, df_test_feats_global):
    """
    處理特定fold的縮放和編碼
    假設輸入的DataFrame包含：
    - ORIGINAL_NUM_FEATS的原始數值列
    - 原始分類/聚類列：'video_category_raw_val', 'text_cluster_raw_val', etc.
    - 預先提取的文本、視覺、音頻embeddings
    """
    log_print("Preprocessing data for a new fold...")
    X_train_parts, X_val_parts, X_test_parts = [], [], []

    # 數值特徵縮放
    scaler = StandardScaler()
    num_train_scaled = scaler.fit_transform(df_train_fold_feats[ORIGINAL_NUM_FEATS])
    num_val_scaled = scaler.transform(df_val_fold_feats[ORIGINAL_NUM_FEATS])
    num_test_scaled = scaler.transform(df_test_feats_global[ORIGINAL_NUM_FEATS])
    
    X_train_parts.append(pd.DataFrame(num_train_scaled, columns=ORIGINAL_NUM_FEATS, index=df_train_fold_feats.index))
    X_val_parts.append(pd.DataFrame(num_val_scaled, columns=ORIGINAL_NUM_FEATS, index=df_val_fold_feats.index))
    X_test_parts.append(pd.DataFrame(num_test_scaled, columns=ORIGINAL_NUM_FEATS, index=df_test_feats_global.index))
    
    scalers_and_encoders = {'scaler': scaler}

    # 分類特徵編碼
    cat_col_map = {
        'video_category_raw_val': 'cat_id',
        'text_cluster_raw_val': 'text_cluster_id',
        'visual_cluster_raw_val': 'visual_cluster_id',
        'user_cluster_raw_val': 'user_cluster_id',
        'audio_cluster_raw_val': 'audio_cluster_id'
    }
    
    fill_values = {
        'video_category_raw_val': "Unknown",
        'text_cluster_raw_val': -1,
        'visual_cluster_raw_val': -1,
        'user_cluster_raw_val': -1,
        'audio_cluster_raw_val': -1
    }

    encoded_cat_names_for_lgbm = []

    for raw_col, encoded_col_name in cat_col_map.items():
        if raw_col not in df_train_fold_feats.columns:
            log_print(f"Warning: {raw_col} not found in training data. Skipping.")
            continue
            
        le = LabelEncoder()
        fill_val = fill_values[raw_col]
        train_series_to_fit = df_train_fold_feats[raw_col].fillna(fill_val)
        le.fit(train_series_to_fit)

        # 訓練集編碼
        X_train_parts.append(
            pd.Series(le.transform(train_series_to_fit), name=encoded_col_name, index=df_train_fold_feats.index).to_frame()
        )
        
        # 驗證集編碼
        val_series_to_transform = df_val_fold_feats[raw_col] if raw_col in df_val_fold_feats.columns else pd.Series(fill_val, index=df_val_fold_feats.index)
        X_val_parts.append(
            safe_label_encode(val_series_to_transform, le, encoded_col_name, 
                            fill_unknown_value=fill_val, unknown_value_in_encoder_fit=fill_val).rename(encoded_col_name).to_frame()
        )

        # 測試集編碼
        test_series_to_transform = df_test_feats_global[raw_col] if raw_col in df_test_feats_global.columns else pd.Series(fill_val, index=df_test_feats_global.index)
        X_test_parts.append(
            safe_label_encode(test_series_to_transform, le, encoded_col_name, 
                            fill_unknown_value=fill_val, unknown_value_in_encoder_fit=fill_val).rename(encoded_col_name).to_frame()
        )

        scalers_and_encoders[f'le_{encoded_col_name}'] = le
        encoded_cat_names_for_lgbm.append(encoded_col_name)

    # 添加預先提取的文本、視覺、音頻embeddings
    text_vision_audio_cols = [col for col in df_train_fold_feats.columns 
                             if col.startswith('text_emb_') or col.startswith('vision_emb_') or col.startswith('audio_emb_')]

    X_train_parts.append(df_train_fold_feats[text_vision_audio_cols])
    X_val_parts.append(df_val_fold_feats[text_vision_audio_cols])
    X_test_parts.append(df_test_feats_global[text_vision_audio_cols])

    # 合併所有特徵
    X_train_fold = pd.concat(X_train_parts, axis=1)
    X_val_fold = pd.concat(X_val_parts, axis=1)
    X_test_fold = pd.concat(X_test_parts, axis=1)
    
    # 確保測試集列與訓練集列匹配
    missing_cols_in_test = set(X_train_fold.columns) - set(X_test_fold.columns)
    for c in missing_cols_in_test:
        log_print(f"Warning: Column '{c}' from training missing in test. Adding as zeros.")
        X_test_fold[c] = 0
        
    extra_cols_in_test = set(X_test_fold.columns) - set(X_train_fold.columns)
    if extra_cols_in_test:
        log_print(f"Warning: Extra columns in test: {extra_cols_in_test}. Dropping.")
        X_test_fold = X_test_fold.drop(columns=list(extra_cols_in_test), errors='ignore')
    
    X_test_fold = X_test_fold[X_train_fold.columns]

    log_print(f"Fold preprocessing complete. Train: {X_train_fold.shape}, Val: {X_val_fold.shape}, Test: {X_test_fold.shape}")
    return X_train_fold, X_val_fold, X_test_fold, scalers_and_encoders, encoded_cat_names_for_lgbm

# ---------- 3. Cross-Validation Loop ----------
log_print("="*60)
log_print("STEP 3: Starting Cross-Validation Loop")
log_print("="*60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_predictions_test = []
fold_mae_scores = []
fold_r2_scores = []
fold_mse_scores = []

fold_predictions_val = []  # 存儲每個fold的驗證預測
fold_ground_truths_val = []  # 存儲每個fold的驗證真實值
fold_calibrators = []  # 存儲每個fold的校正器

for fold_idx, (train_indices, val_indices) in enumerate(kf.split(df_train_full_features, y_train_full)):
    fold_start_time = time.time()
    current_fold_output_dir = os.path.join(BASE_OUTPUT_DIR, f"fold_{fold_idx+1}")
    os.makedirs(current_fold_output_dir, exist_ok=True)
    log_print(f"\n===== Starting Fold {fold_idx+1}/{N_FOLDS} =====")
    log_print(f"Fold output directory: {current_fold_output_dir}")

    # 切分當前fold的數據
    df_train_fold_current_feats = df_train_full_features.iloc[train_indices].copy()
    y_train_fold_current = y_train_full.iloc[train_indices].copy()
    df_val_fold_current_feats = df_train_full_features.iloc[val_indices].copy()
    y_val_fold_current = y_train_full.iloc[val_indices].copy()

    log_print(f"Fold {fold_idx+1} - Train size: {len(train_indices)}, Val size: {len(val_indices)}")

    # 每個fold的預處理（縮放、編碼）
    X_train_fold_processed, X_val_fold_processed, X_test_fold_processed, \
    fold_preprocessors, fold_cat_feat_names = preprocess_data_for_fold(
        df_train_fold_current_feats, df_val_fold_current_feats, df_test_full_features.copy()
    )

    # 保存當前fold的預處理器
    for name, proc in fold_preprocessors.items():
        joblib.dump(proc, os.path.join(current_fold_output_dir, f"preproc_{name}.pkl"))

    # LightGBM模型訓練
    log_print(f"Fold {fold_idx+1} - Starting LightGBM model training...")
    lgb_train_fold_data = lgb.Dataset(
        X_train_fold_processed, y_train_fold_current,
        feature_name=list(X_train_fold_processed.columns),
        categorical_feature=fold_cat_feat_names if fold_cat_feat_names else 'auto'
    )
    lgb_eval_fold_data = lgb.Dataset(
        X_val_fold_processed, y_val_fold_current, 
        reference=lgb_train_fold_data,
        feature_name=list(X_val_fold_processed.columns),
        categorical_feature=fold_cat_feat_names if fold_cat_feat_names else 'auto'
    )

    lgbm_fold_params = {
        'objective': 'regression_l1', 
        'metric': ['l1', 'l2'], 
        'boosting_type': 'gbdt',
        'num_leaves': 31, 
        'learning_rate': 0.05, 
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 
        'bagging_freq': 5, 
        'verbose': -1, 
        'n_jobs': -1,
        'seed': SEED + fold_idx,  # 每個fold略微不同的種子
        'num_threads': os.cpu_count() if os.cpu_count() else 4,
    }
    
    lgbm_fold_callbacks = [
        lgb.log_evaluation(period=200, show_stdv=True),
        lgb.early_stopping(stopping_rounds=100, verbose=True)
    ]

    model_fold = lgb.train(
        lgbm_fold_params, lgb_train_fold_data,
        num_boost_round=2000, 
        valid_sets=[lgb_train_fold_data, lgb_eval_fold_data],
        valid_names=['train', 'eval'], 
        callbacks=lgbm_fold_callbacks
    )
    
    log_print(f"Fold {fold_idx+1} - LightGBM training completed.")
    model_fold.save_model(os.path.join(current_fold_output_dir, "lightgbm_model_fold.txt"))

    # 在驗證集上評估
    y_pred_val_fold = model_fold.predict(X_val_fold_processed, num_iteration=model_fold.best_iteration)
    
    # 保存原始驗證預測和真實值
    fold_predictions_val.append(y_pred_val_fold.copy())
    fold_ground_truths_val.append(y_val_fold_current.copy())
    
    # 訓練校正器
    log_print(f"Fold {fold_idx+1} - Training popularity calibrator...")
    calibrator = PopularityCalibrator(method='ensemble')
    calibrator.fit(y_pred_val_fold, y_val_fold_current)
    fold_calibrators.append(calibrator)
    
    # 應用校正到驗證預測
    y_pred_val_calibrated = calibrator.transform(y_pred_val_fold)
    
    # 計算校正前後的指標
    mae_val_fold_original = mean_absolute_error(y_val_fold_current, y_pred_val_fold)
    mae_val_fold_calibrated = mean_absolute_error(y_val_fold_current, y_pred_val_calibrated)
    r2_val_fold_original = r2_score(y_val_fold_current, y_pred_val_fold)
    r2_val_fold_calibrated = r2_score(y_val_fold_current, y_pred_val_calibrated)
    mse_val_fold_calibrated = mean_squared_error(y_val_fold_current, y_pred_val_calibrated)
    
    # 🔧 統一變數名稱（用於後續繪圖）
    mae_val_fold = mae_val_fold_calibrated
    r2_val_fold = r2_val_fold_calibrated
    mse_val_fold = mse_val_fold_calibrated
    
    log_print(f"Fold {fold_idx+1} - Calibration Results:")
    log_print(f"  Original:   MAE={mae_val_fold_original:.4f}, R2={r2_val_fold_original:.4f}")
    log_print(f"  Calibrated: MAE={mae_val_fold_calibrated:.4f}, R2={r2_val_fold_calibrated:.4f}")
    log_print(f"  Improvement: MAE Δ={mae_val_fold_original-mae_val_fold_calibrated:+.4f}, R2 Δ={r2_val_fold_calibrated-r2_val_fold_original:+.4f}")
    
    # 更新fold指標記錄（使用校正後的值）
    fold_mae_scores.append(mae_val_fold_calibrated)
    fold_r2_scores.append(r2_val_fold_calibrated)
    fold_mse_scores.append(mse_val_fold_calibrated)
    
    # 🔧 關鍵修復：在測試集上預測並應用校正
    log_print(f"Fold {fold_idx+1} - Performing inference on test data...")
    y_pred_test_fold_original = model_fold.predict(X_test_fold_processed, num_iteration=model_fold.best_iteration)
    
    log_print(f"Fold {fold_idx+1} - Applying calibration to test predictions...")
    y_pred_test_fold_calibrated = calibrator.transform(y_pred_test_fold_original)
    
    # 🔧 重要：存儲校正後的測試預測
    fold_predictions_test.append(y_pred_test_fold_calibrated)
    
    # 🔧 調試信息：檢查校正是否真的在工作
    log_print(f"Fold {fold_idx+1} - Test prediction verification:")
    log_print(f"  Original[0:3]:   {y_pred_test_fold_original[:3]}")
    log_print(f"  Calibrated[0:3]: {y_pred_test_fold_calibrated[:3]}")
    log_print(f"  Mean difference: {(y_pred_test_fold_calibrated - y_pred_test_fold_original).mean():.4f}")
    log_print(f"  Are they equal?: {np.array_equal(y_pred_test_fold_original, y_pred_test_fold_calibrated)}")
    
    # 保存校正器
    joblib.dump(calibrator, os.path.join(current_fold_output_dir, "popularity_calibrator.pkl"))
    
    # 更新fold指標記錄，添加校正相關信息
    fold_metrics = {
        "fold": fold_idx+1, 
        "val_mae_original": mae_val_fold_original,
        "val_mae_calibrated": mae_val_fold_calibrated,
        "val_r2_original": r2_val_fold_original,
        "val_r2_calibrated": r2_val_fold_calibrated,
        "mae_improvement": mae_val_fold_original - mae_val_fold_calibrated,
        "r2_improvement": r2_val_fold_calibrated - r2_val_fold_original,
        "best_iteration": model_fold.best_iteration,
        "audio_feature_dim": AUDIO_FEAT_DIM
    }
    
    with open(os.path.join(current_fold_output_dir, "fold_eval_metrics.json"), 'w') as f_json:
        json.dump(fold_metrics, f_json, indent=4)
    
    # 繪製校正前後對比圖
    plt.figure(figsize=(15, 5))
    
    # 原始預測
    plt.subplot(1, 3, 1)
    plt.scatter(y_val_fold_current, y_pred_val_fold, alpha=0.6, color='blue')
    min_val = min(y_val_fold_current.min(), y_pred_val_fold.min())
    max_val = max(y_val_fold_current.max(), y_pred_val_fold.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    plt.xlabel('Ground Truth')
    plt.ylabel('Original Prediction')
    plt.title(f'Fold {fold_idx+1}: Before Calibration\nMAE: {mae_val_fold_original:.3f}, R²: {r2_val_fold_original:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 校正後預測
    plt.subplot(1, 3, 2)
    plt.scatter(y_val_fold_current, y_pred_val_calibrated, alpha=0.6, color='green')
    min_val = min(y_val_fold_current.min(), y_pred_val_calibrated.min())
    max_val = max(y_val_fold_current.max(), y_pred_val_calibrated.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    plt.xlabel('Ground Truth')
    plt.ylabel('Calibrated Prediction')
    plt.title(f'Fold {fold_idx+1}: After Calibration\nMAE: {mae_val_fold:.3f}, R²: {r2_val_fold:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 殘差分析
    plt.subplot(1, 3, 3)
    residuals_original = y_val_fold_current - y_pred_val_fold
    residuals_calibrated = y_val_fold_current - y_pred_val_calibrated
    plt.scatter(y_pred_val_fold, residuals_original, alpha=0.6, color='blue', label='Original')
    plt.scatter(y_pred_val_calibrated, residuals_calibrated, alpha=0.6, color='green', label='Calibrated')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Pred)')
    plt.title(f'Fold {fold_idx+1}: Residuals Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(current_fold_output_dir, f"fold_{fold_idx+1}_calibration_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 繪製校正函數圖（顯示原始預測如何映射到校正預測）
    plt.figure(figsize=(10, 6))
    pred_range = np.linspace(y_pred_val_fold.min(), y_pred_val_fold.max(), 100)
    calibrated_range = calibrator.transform(pred_range)
    
    plt.scatter(y_pred_val_fold, y_pred_val_calibrated, alpha=0.6, color='blue', s=20, label='Validation Data')
    plt.plot(pred_range, calibrated_range, 'r-', lw=3, label='Calibration Function')
    plt.plot(pred_range, pred_range, 'k--', lw=2, alpha=0.7, label='Identity Line')
    plt.xlabel('Original Prediction')
    plt.ylabel('Calibrated Prediction')
    plt.title(f'Fold {fold_idx+1}: Calibration Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(current_fold_output_dir, f"fold_{fold_idx+1}_calibration_function.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🔧 保存多版本的測試預測進行對比
    if pids_test is not None and len(pids_test) == len(y_pred_test_fold_calibrated):
        # 1. 原始預測版本
        submission_df_fold_original = pd.DataFrame({
            'pid': pids_test, 
            'popularity_score': y_pred_test_fold_original
        })
        submission_df_fold_original['pid'] = submission_df_fold_original['pid'].astype(str)
        submission_df_fold_original.sort_values(by="pid", inplace=True)
        fold_submission_path_original = os.path.join(current_fold_output_dir, f"test_predictions_fold_{fold_idx+1}_original.csv")
        submission_df_fold_original.to_csv(fold_submission_path_original, index=False)
        
        # 2. 校正後預測版本（這是主要的輸出）
        submission_df_fold_calibrated = pd.DataFrame({
            'pid': pids_test, 
            'popularity_score': y_pred_test_fold_calibrated
        })
        submission_df_fold_calibrated['pid'] = submission_df_fold_calibrated['pid'].astype(str)
        submission_df_fold_calibrated.sort_values(by="pid", inplace=True)
        fold_submission_path_calibrated = os.path.join(current_fold_output_dir, f"test_predictions_fold_{fold_idx+1}.csv")
        submission_df_fold_calibrated.to_csv(fold_submission_path_calibrated, index=False)
        
        # 3. 對比版本
        submission_df_fold_comparison = pd.DataFrame({
            'pid': pids_test,
            'popularity_score_original': y_pred_test_fold_original,
            'popularity_score_calibrated': y_pred_test_fold_calibrated,
            'calibration_adjustment': y_pred_test_fold_calibrated - y_pred_test_fold_original
        })
        submission_df_fold_comparison['pid'] = submission_df_fold_comparison['pid'].astype(str)
        submission_df_fold_comparison.sort_values(by="pid", inplace=True)
        fold_submission_path_comparison = os.path.join(current_fold_output_dir, f"test_predictions_fold_{fold_idx+1}_comparison.csv")
        submission_df_fold_comparison.to_csv(fold_submission_path_comparison, index=False)
        
        log_print(f"Fold {fold_idx+1} - Test predictions saved:")
        log_print(f"  Original:   {fold_submission_path_original}")
        log_print(f"  Calibrated: {fold_submission_path_calibrated}")
        log_print(f"  Comparison: {fold_submission_path_comparison}")
        
        # 🔧 統計信息驗證
        log_print(f"Fold {fold_idx+1} - Test prediction statistics:")
        log_print(f"  Original   - Mean: {y_pred_test_fold_original.mean():.4f}, Std: {y_pred_test_fold_original.std():.4f}")
        log_print(f"  Calibrated - Mean: {y_pred_test_fold_calibrated.mean():.4f}, Std: {y_pred_test_fold_calibrated.std():.4f}")
        log_print(f"  Adjustment - Mean: {(y_pred_test_fold_calibrated - y_pred_test_fold_original).mean():.4f}")
        
    else:
        log_print(f"Fold {fold_idx+1} - Error saving test predictions (pid/length mismatch).")

    # 繪製當前fold的GT vs Pred圖（使用校正後的預測）
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_val_fold_current, y=y_pred_val_calibrated, alpha=0.6)
    min_val = min(y_val_fold_current.min(), y_pred_val_calibrated.min())
    max_val = max(y_val_fold_current.max(), y_pred_val_calibrated.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Ground Truth Popularity')
    plt.ylabel('Predicted Popularity (Calibrated)')
    plt.title(f'Fold {fold_idx+1}: GT vs Pred (Calibrated) - MAE: {mae_val_fold:.3f}, R2: {r2_val_fold:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(current_fold_output_dir, f"fold_{fold_idx+1}_gt_vs_pred_calibrated.png"))
    plt.close()

    # 特徵重要性圖
    plt.figure(figsize=(12, max(8, len(X_train_fold_processed.columns) // 4)))
    lgb.plot_importance(model_fold, max_num_features=min(30, len(X_train_fold_processed.columns)), 
                       height=0.5, importance_type='gain', 
                       title=f"Fold {fold_idx+1} - Feature Importance (Top 30)")
    plt.tight_layout()
    plt.savefig(os.path.join(current_fold_output_dir, f"fold_{fold_idx+1}_feature_importance.png"))
    plt.close()

    fold_time = (time.time() - fold_start_time) / 60
    log_print(f"Fold {fold_idx+1} completed in {fold_time:.2f} minutes.")
    
    # 清理內存
    del X_train_fold_processed, X_val_fold_processed, X_test_fold_processed
    del df_train_fold_current_feats, df_val_fold_current_feats
    del y_train_fold_current, y_val_fold_current
    del y_pred_val_fold, y_pred_val_calibrated
    del y_pred_test_fold_original, y_pred_test_fold_calibrated
    gc.collect()

# ---------- 4. Ensemble Test Predictions and Final Analysis ----------
log_print("="*60)
log_print("STEP 4: Enhanced Ensemble with Calibrated Predictions")
log_print("="*60)

log_print("\n===== Analyzing Calibration Effectiveness Across Folds =====")
overall_mae_improvement = []
overall_r2_improvement = []

for i in range(N_FOLDS):
    original_mae = mean_absolute_error(fold_ground_truths_val[i], fold_predictions_val[i])
    calibrated_pred = fold_calibrators[i].transform(fold_predictions_val[i])
    calibrated_mae = mean_absolute_error(fold_ground_truths_val[i], calibrated_pred)
    
    original_r2 = r2_score(fold_ground_truths_val[i], fold_predictions_val[i])
    calibrated_r2 = r2_score(fold_ground_truths_val[i], calibrated_pred)
    
    mae_improvement = original_mae - calibrated_mae
    r2_improvement = calibrated_r2 - original_r2
    
    overall_mae_improvement.append(mae_improvement)
    overall_r2_improvement.append(r2_improvement)
    
    log_print(f"Fold {i+1}: MAE improvement: {mae_improvement:+.4f}, R² improvement: {r2_improvement:+.4f}")

log_print(f"\nAverage MAE improvement across folds: {np.mean(overall_mae_improvement):+.4f} ± {np.std(overall_mae_improvement):.4f}")
log_print(f"Average R² improvement across folds: {np.mean(overall_r2_improvement):+.4f} ± {np.std(overall_r2_improvement):.4f}")

# 集成測試預測（已經是校正後的預測）
log_print("\n===== Creating Ensemble from Calibrated Predictions =====")
if len(fold_predictions_test) == N_FOLDS and pids_test is not None:
    if all(len(pred_arr) == len(pids_test) for pred_arr in fold_predictions_test):
        # 計算集成統計
        avg_test_predictions = np.mean(np.array(fold_predictions_test), axis=0)
        std_test_predictions = np.std(np.array(fold_predictions_test), axis=0)
        median_test_predictions = np.median(np.array(fold_predictions_test), axis=0)
        
        # 創建完整的ensemble submission DataFrame
        ensemble_submission_df = pd.DataFrame({
            'pid': pids_test,
            'popularity_score': avg_test_predictions,
            'prediction_std': std_test_predictions,
            'prediction_median': median_test_predictions,
            'prediction_min': np.min(np.array(fold_predictions_test), axis=0),
            'prediction_max': np.max(np.array(fold_predictions_test), axis=0)
        })
        ensemble_submission_df['pid'] = ensemble_submission_df['pid'].astype(str)
        ensemble_submission_df.sort_values(by="pid", inplace=True)
        
        # 保存不同版本的預測
        # 1. 標準提交版本（平均值）
        ensemble_submission_path = os.path.join(BASE_OUTPUT_DIR, "test_predictions_ensemble_calibrated.csv")
        ensemble_submission_df[['pid', 'popularity_score']].to_csv(ensemble_submission_path, index=False)
        
        # 2. 中位數版本
        ensemble_submission_median_path = os.path.join(BASE_OUTPUT_DIR, "test_predictions_ensemble_median_calibrated.csv")
        ensemble_submission_df[['pid', 'prediction_median']].rename(columns={'prediction_median': 'popularity_score'}).to_csv(ensemble_submission_median_path, index=False)
        
        # 3. 完整版本（包含所有統計信息）
        ensemble_submission_full_path = os.path.join(BASE_OUTPUT_DIR, "test_predictions_ensemble_full_calibrated.csv")
        ensemble_submission_df.to_csv(ensemble_submission_full_path, index=False)
        
        # 4. 最終提交版本
        final_submission_path = os.path.join(BASE_OUTPUT_DIR, "FINAL_test_predictions_CALIBRATED.csv")
        ensemble_submission_df[['pid', 'popularity_score']].to_csv(final_submission_path, index=False)
        
        log_print(f"Calibrated ensemble predictions saved:")
        log_print(f"  Standard (mean): {ensemble_submission_path}")
        log_print(f"  Median version: {ensemble_submission_median_path}")
        log_print(f"  Full statistics: {ensemble_submission_full_path}")
        log_print(f"  FINAL SUBMISSION: {final_submission_path}")
        log_print(f"Ensemble DataFrame head:\n{ensemble_submission_df.head()}")
        
        # 預測統計分析
        log_print(f"\nCalibrated Prediction Statistics:")
        log_print(f"  Mean: {avg_test_predictions.mean():.4f}")
        log_print(f"  Std: {avg_test_predictions.std():.4f}")
        log_print(f"  Min: {avg_test_predictions.min():.4f}")
        log_print(f"  Max: {avg_test_predictions.max():.4f}")
        log_print(f"  Average prediction uncertainty (std): {std_test_predictions.mean():.4f}")
        log_print(f"  Median prediction uncertainty: {np.median(std_test_predictions):.4f}")
        
        # 分析高/低不確定性的預測
        high_uncertainty_mask = std_test_predictions > np.percentile(std_test_predictions, 95)
        low_uncertainty_mask = std_test_predictions < np.percentile(std_test_predictions, 5)
        
        log_print(f"\nPrediction Uncertainty Analysis:")
        log_print(f"  High uncertainty samples (top 5%): {np.sum(high_uncertainty_mask)} samples")
        log_print(f"    Average prediction: {avg_test_predictions[high_uncertainty_mask].mean():.4f}")
        log_print(f"    Average uncertainty: {std_test_predictions[high_uncertainty_mask].mean():.4f}")
        log_print(f"  Low uncertainty samples (bottom 5%): {np.sum(low_uncertainty_mask)} samples")
        log_print(f"    Average prediction: {avg_test_predictions[low_uncertainty_mask].mean():.4f}")
        log_print(f"    Average uncertainty: {std_test_predictions[low_uncertainty_mask].mean():.4f}")
        
    else:
        log_print("Error: Prediction arrays from folds have inconsistent lengths. Cannot ensemble.")
else:
    log_print("Error: Not enough fold predictions to ensemble or pids_test is missing.")

# Cross-Validation 總結(使用校正後的指標)
log_print(f"\n===== Enhanced Cross-Validation Summary (with Calibration) =====")
log_print(f"Validation MAEs (calibrated): {fold_mae_scores}")
log_print(f"Validation R2s (calibrated): {fold_r2_scores}")
log_print(f"Mean Validation MAE (calibrated): {np.mean(fold_mae_scores):.4f} (+/- {np.std(fold_mae_scores):.4f})")
log_print(f"Mean Validation R2 (calibrated): {np.mean(fold_r2_scores):.4f} (+/- {np.std(fold_r2_scores):.4f})")
log_print(f"Average MAE improvement from calibration: {np.mean(overall_mae_improvement):+.4f}")
log_print(f"Average R² improvement from calibration: {np.mean(overall_r2_improvement):+.4f}")

# 保存CV指標
cv_summary_metrics = {
    "mean_val_mae_calibrated": float(np.mean(fold_mae_scores)),
    "std_val_mae_calibrated": float(np.std(fold_mae_scores)),
    "mean_val_r2_calibrated": float(np.mean(fold_r2_scores)),
    "std_val_r2_calibrated": float(np.std(fold_r2_scores)),
    "mean_mae_improvement_from_calibration": float(np.mean(overall_mae_improvement)),
    "std_mae_improvement_from_calibration": float(np.std(overall_mae_improvement)),
    "mean_r2_improvement_from_calibration": float(np.mean(overall_r2_improvement)),
    "std_r2_improvement_from_calibration": float(np.std(overall_r2_improvement)),
    "individual_fold_maes_calibrated": fold_mae_scores,
    "individual_fold_r2s_calibrated": fold_r2_scores,
    "individual_mae_improvements": overall_mae_improvement,
    "individual_r2_improvements": overall_r2_improvement,
    "audio_feature_dim": AUDIO_FEAT_DIM,
    "n_folds": N_FOLDS,
    "calibration_method": "ensemble"
}

with open(os.path.join(BASE_OUTPUT_DIR, "cv_summary_metrics_calibrated.json"), 'w') as f_json:
    json.dump(cv_summary_metrics, f_json, indent=4)

# 創建簡化的可視化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 第一行：性能指標
# MAE對比（校正前後）
if len(overall_mae_improvement) > 0:
    original_maes = [fold_mae_scores[i] + overall_mae_improvement[i] for i in range(N_FOLDS)]
    x_pos = range(1, N_FOLDS + 1)
    width = 0.35
    
    axes[0, 0].bar([x - width/2 for x in x_pos], original_maes, width, 
                   label='Original', color='lightcoral', alpha=0.7)
    axes[0, 0].bar([x + width/2 for x in x_pos], fold_mae_scores, width,
                   label='Calibrated', color='lightblue', alpha=0.7)
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Validation MAE')
    axes[0, 0].set_title('MAE: Before vs After Calibration')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

# R²對比
if len(overall_r2_improvement) > 0:
    original_r2s = [fold_r2_scores[i] - overall_r2_improvement[i] for i in range(N_FOLDS)]
    axes[0, 1].bar([x - width/2 for x in x_pos], original_r2s, width,
                   label='Original', color='lightcoral', alpha=0.7)
    axes[0, 1].bar([x + width/2 for x in x_pos], fold_r2_scores, width,
                   label='Calibrated', color='lightgreen', alpha=0.7)
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Validation R²')
    axes[0, 1].set_title('R²: Before vs After Calibration')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# 改進幅度分布
axes[0, 2].scatter(overall_mae_improvement, overall_r2_improvement, s=100, alpha=0.7)
for i, (mae_imp, r2_imp) in enumerate(zip(overall_mae_improvement, overall_r2_improvement)):
    axes[0, 2].annotate(f'F{i+1}', (mae_imp, r2_imp), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0, 2].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[0, 2].set_xlabel('MAE Improvement')
axes[0, 2].set_ylabel('R² Improvement')
axes[0, 2].set_title('Calibration Improvement by Fold')
axes[0, 2].grid(True, alpha=0.3)

# 第二行：預測分析
# 預測不確定性分布
if 'std_test_predictions' in locals():
    axes[1, 0].hist(std_test_predictions, bins=30, color='orange', alpha=0.7)
    axes[1, 0].axvline(x=np.mean(std_test_predictions), color='red', linestyle='--',
                       label=f'Mean: {np.mean(std_test_predictions):.4f}')
    axes[1, 0].set_xlabel('Prediction Standard Deviation')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Test Prediction Uncertainty Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'No uncertainty data\navailable', 
                    ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Prediction Uncertainty')

# 預測值分布
if 'avg_test_predictions' in locals():
    axes[1, 1].hist(avg_test_predictions, bins=30, color='skyblue', alpha=0.7)
    axes[1, 1].axvline(x=np.mean(avg_test_predictions), color='red', linestyle='--',
                       label=f'Mean: {np.mean(avg_test_predictions):.4f}')
    axes[1, 1].set_xlabel('Predicted Popularity Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Test Prediction Distribution (Calibrated)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'No prediction data\navailable', 
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Prediction Distribution')

# 預測值 vs 不確定性散點圖
if 'avg_test_predictions' in locals() and 'std_test_predictions' in locals():
    scatter = axes[1, 2].scatter(avg_test_predictions, std_test_predictions, 
                                alpha=0.6, c=avg_test_predictions, cmap='viridis', s=20)
    axes[1, 2].set_xlabel('Predicted Popularity Score')
    axes[1, 2].set_ylabel('Prediction Uncertainty (Std)')
    axes[1, 2].set_title('Prediction vs Uncertainty')
    plt.colorbar(scatter, ax=axes[1, 2], label='Predicted Score')
    axes[1, 2].grid(True, alpha=0.3)
else:
    axes[1, 2].text(0.5, 0.5, 'No data available\nfor analysis', 
                    ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Prediction vs Uncertainty')

plt.tight_layout()
plt.savefig(os.path.join(BASE_OUTPUT_DIR, "cv_summary_plots_with_calibration.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 最終摘要
total_script_time_minutes = (time.time() - script_start_time) / 60
log_print("\n" + "="*60)
log_print("ENHANCED EXECUTION SUMMARY (WITH CALIBRATION)")
log_print("="*60)
log_print(f"Training data shape: {df_train_full_features.shape}")
log_print(f"Test data shape: {df_test_full_features.shape}")
log_print(f"Audio feature dimension: {AUDIO_FEAT_DIM}")
log_print(f"Number of folds: {N_FOLDS}")
log_print(f"Calibration method: ensemble")
log_print(f"Mean CV MAE (calibrated): {np.mean(fold_mae_scores):.4f} ± {np.std(fold_mae_scores):.4f}")
log_print(f"Mean CV R² (calibrated): {np.mean(fold_r2_scores):.4f} ± {np.std(fold_r2_scores):.4f}")
log_print(f"Average MAE improvement: {np.mean(overall_mae_improvement):+.4f} ± {np.std(overall_mae_improvement):.4f}")
log_print(f"Average R² improvement: {np.mean(overall_r2_improvement):+.4f} ± {np.std(overall_r2_improvement):.4f}")

if 'avg_test_predictions' in locals():
    log_print(f"Calibrated test predictions range: [{avg_test_predictions.min():.4f}, {avg_test_predictions.max():.4f}]")
    log_print(f"Average prediction uncertainty: {std_test_predictions.mean():.4f}")

log_print(f"Total execution time: {total_script_time_minutes:.2f} minutes")
log_print(f"Output directory: {os.path.abspath(BASE_OUTPUT_DIR)}")

if 'final_submission_path' in locals():
    log_print(f"\nFINAL SUBMISSION FILE: {final_submission_path}")

log_print("="*60)
log_print("LightGBM Enhanced Cross-Validation with Calibration COMPLETED!")
log_print("="*60)