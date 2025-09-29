import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelValidator:
    def __init__(self, model, feature_engineer):
        self.model = model
        self.feature_engineer = feature_engineer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def predict_single(self, t1, h1, t2, t):
        """单个预测"""
        # 准备输入特征
        input_data = {
            't1': [t1], 'h1': [h1], 't2': [t2], 't': [t],
            'h2_measured': [h1]  # 初始时传感器读数等于初始湿度
        }
        df = pd.DataFrame(input_data)
        
        # 应用特征工程
        df_features = self.feature_engineer.create_features(df)
        available_features = [col for col in self.feature_engineer.scaler.feature_names_in_ 
                            if col in df_features.columns]
        X = df_features[available_features].values
        X_scaled = self.feature_engineer.scaler.transform(X)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            prediction = self.model(X_tensor.unsqueeze(1))
            return prediction.item()
    
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor.unsqueeze(1)).cpu().numpy().flatten()
        
        # 计算评估指标
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("模型评估结果:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f}")
        
        return predictions, {'mse': mse, 'mae': mae, 'r2': r2}
    
    def cross_environment_validation(self, test_data):
        """跨环境验证"""
        environments = test_data.groupby(['t1', 't2']).size().reset_index()
        results = []
        
        for _, env in environments.iterrows():
            env_data = test_data[(test_data['t1'] == env['t1']) & (test_data['t2'] == env['t2'])]
            if len(env_data) < 5:  # 至少需要5个样本
                continue
                
            X_env = env_data.drop(['h2_actual'], axis=1, errors='ignore').values
            y_env = env_data['h2_actual'].values if 'h2_actual' in env_data.columns else env_data['h2_measured'].values
            
            # 应用特征工程
            X_env_scaled = self.feature_engineer.scaler.transform(X_env)
            
            # 评估
            predictions, metrics = self.evaluate_model(X_env_scaled, y_env)
            results.append({
                'environment': f"t1={env['t1']}, t2={env['t2']}",
                'samples': len(env_data),
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'r2': metrics['r2']
            })
        
        # 显示跨环境结果
        print("\n跨环境验证结果:")
        for result in results:
            print(f"  {result['environment']} (n={result['samples']}): "
                  f"MSE={result['mse']:.4f}, MAE={result['mae']:.4f}, R²={result['r2']:.4f}")
        
        return results
    
    def plot_predictions(self, X_test, y_test, predictions, title="Model Predictions"):
        """绘制预测结果"""
        plt.figure(figsize=(12, 5))
        
        # 预测vs真实值散点图
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs True Values')
        
        # 残差图
        plt.subplot(1, 2, 2)
        residuals = y_test - predictions
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()
    
    def stress_test(self):
        """压力测试：极端条件下的模型表现"""
        extreme_conditions = [
            {'name': '高温高湿', 't1': 10, 'h1': 20, 't2': 40, 't': 30},
            {'name': '低温低湿', 't1': 40, 'h1': 80, 't2': 10, 't': 30},
            {'name': '大温差', 't1': 15, 'h1': 50, 't2': 35, 't': 10},
            {'name': '小温差', 't1': 25, 'h1': 45, 't2': 26, 't': 45},
        ]
        
        print("\n压力测试结果:")
        for condition in extreme_conditions:
            pred = self.predict_single(
                condition['t1'], condition['h1'], 
                condition['t2'], condition['t']
            )
            print(f"  {condition['name']}: 预测h2 = {pred:.2f}%RH")

# 完整的验证流程
def complete_validation():
    # 加载数据
    df = pd.read_csv('real_sensor_data.csv')
    
    # 特征工程
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.prepare_data_for_training(df)
    
    # 分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = train_with_tuning(X_train, y_train)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('tuned_sensor_transformer.pth'))
    
    # 验证模型
    validator = ModelValidator(model, engineer)
    
    # 基本评估
    predictions, metrics = validator.evaluate_model(X_test, y_test)
    
    # 绘制结果
    validator.plot_predictions(X_test, y_test, predictions)
    
    # 跨环境验证
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['h2_actual'] = y_test
    validator.cross_environment_validation(df_test)
    
    # 压力测试
    validator.stress_test()
    
    return model, validator

# 运行完整验证
# model, validator = complete_validation()