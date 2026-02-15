import pickle
import xgboost as xgb

# 1. 載入模型
with open('model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. 問它：你訓練時到底看了哪些欄位？
print("\n=== XGBoost 嚴格要求的欄位順序 (請複製這個列表) ===")
# XGBoost 儲存欄位名稱的方式比較特別
print(model.get_booster().feature_names)
print("===================================================\n")