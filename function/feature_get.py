import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def create_features(self, df):
        """创建新特征"""
        df_new = df.copy()
        
        # 基础特征
        df_new['delta_t'] = df_new['t2'] - df_new['t1']  # 温度差
        df_new['delta_h_initial'] = df_new['h2_measured'] - df_new['h1']  # 湿度变化
        df_new['t_normalized'] = df_new['t'] / (df_new['t'] + 1)  # 时间归一化
        df_new['temp_ratio'] = df_new['t2'] / (df_new['t1'] + 273.15)  # 温度比（开尔文）
        
        # 复合特征
        df_new['temp_humidity_interaction'] = df_new['delta_t'] * df_new['h1']
        df_new['time_temp_interaction'] = df_new['t'] * df_new['delta_t']
        
        # 指数特征（基于物理模型）
        df_new['exp_factor'] = np.exp(-df_new['t'] / 30)  # 假设典型时间常数为30秒
        df_new['log_time'] = np.log(df_new['t'] + 1)  # 对数时间
        
        # 移动平均特征（如果有时间序列）
        if 'timestamp' in df_new.columns:
            df_new = df_new.sort_values('timestamp')
            df_new['h1_ma5'] = df_new['h1'].rolling(window=5, min_periods=1).mean()
            df_new['t2_ma5'] = df_new['t2'].rolling(window=5, min_periods=1).mean()
        
        return df_new
    
    def select_features(self, X, y, k=10):
        """选择最重要的特征"""
        self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        return X_selected
    
    def get_feature_names(self, original_features):
        """获取选中特征的名称"""
        if self.feature_selector:
            selected_indices = self.feature_selector.get_support(indices=True)
            return [original_features[i] for i in selected_indices]
        return original_features
    
    def prepare_data_for_training(self, df):
        """准备训练数据"""
        # 创建特征
        df_features = self.create_features(df)
        
        # 定义输入特征和目标变量
        feature_columns = [
            't1', 'h1', 't2', 't', 'delta_t', 'delta_h_initial',
            't_normalized', 'temp_ratio', 'temp_humidity_interaction',
            'time_temp_interaction', 'exp_factor', 'log_time'
        ]
        
        # 确保所有特征列都存在
        available_features = [col for col in feature_columns if col in df_features.columns]
        
        X = df_features[available_features].values
        y = df_features['h2_actual'].values if 'h2_actual' in df_features.columns else df_features['h2_measured'].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, available_features

# 使用特征工程
def apply_feature_engineering():
    # 加载数据
    df = pd.read_csv('real_sensor_data.csv')
    
    # 应用特征工程
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.prepare_data_for_training(df)
    
    print(f"原始特征数: {len(feature_names)}")
    print(f"特征名称: {feature_names}")
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    return X, y, engineer

# 运行特征工程
X, y, feature_engineer = apply_feature_engineering()