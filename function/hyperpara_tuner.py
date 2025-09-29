import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import optuna
from optuna.trial import TrialState

class HyperparameterTuner:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def create_model(self, trial, input_dim):
        """根据trial创建模型"""
        # 建议超参数
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        nhead = trial.suggest_categorical('nhead', [2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 6)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        
        # 创建模型
        model = SensorTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        return model, learning_rate
    
    def objective(self, trial):
        """Optuna优化目标函数"""
        # 创建模型
        model, learning_rate = self.create_model(trial, self.X.shape[1])
        
        # 5折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kfold.split(self.X):
            # 分割数据
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # 转换为PyTorch张量
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
            
            # 训练模型
            model_copy = type(model)(**model.__dict__['_modules'])  # 创建模型副本
            # 实际应用中需要重新初始化模型参数
            model_copy = SensorTransformer(
                input_dim=self.X.shape[1],
                d_model=model.d_model,
                nhead=model.nhead,
                num_layers=model.num_layers,
                dropout=model.dropout.p
            )
            
            val_loss = self.train_model(model_copy, X_train_tensor, y_train_tensor, 
                                      X_val_tensor, y_val_tensor, learning_rate, epochs=50)
            cv_scores.append(val_loss)
        
        return np.mean(cv_scores)
    
    def train_model(self, model, X_train, y_train, X_val, y_val, learning_rate, epochs=50):
        """训练单个模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 转移到设备
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train.unsqueeze(1))
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.unsqueeze(1))
            val_loss = criterion(val_outputs, y_val).item()
        
        return val_loss
    
    def optimize(self, n_trials=50):
        """执行超参数优化"""
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print("最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"最佳验证损失: {study.best_value}")
        
        return study.best_params

# 网格搜索调优（简化版）
def grid_search_tuning(X, y):
    """网格搜索超参数调优"""
    from sklearn.model_selection import ParameterGrid
    import itertools
    
    # 定义参数网格
    param_grid = {
        'd_model': [32, 64],
        'nhead': [2, 4],
        'num_layers': [2, 3],
        'dropout': [0.1, 0.3]
    }
    
    best_score = float('inf')
    best_params = None
    
    # 5折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for params in ParameterGrid(param_grid):
        cv_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建模型
            model = SensorTransformer(
                input_dim=X.shape[1],
                d_model=params['d_model'],
                nhead=params['nhead'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
            
            # 简化的训练和验证（实际应用中需要完整训练）
            # 这里只是一个示例框架
            val_loss = 0.0  # 实际应该计算验证损失
            cv_scores.append(val_loss)
        
        avg_score = np.mean(cv_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
    
    print(f"最佳参数: {best_params}")
    print(f"最佳得分: {best_score}")
    return best_params

# 学习率调度和早停
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# 完整的训练函数（包含调优）
def train_with_tuning(X, y, best_params=None):
    """使用最佳参数训练模型"""
    from sklearn.model_selection import train_test_split
    
    # 分割数据
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 如果没有提供最佳参数，使用默认值
    if best_params is None:
        best_params = {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 3,
            'dropout': 0.1
        }
    
    # 创建模型
    model = SensorTransformer(
        input_dim=X.shape[1],
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # 训练设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=15)
    
    # 训练循环
    num_epochs = 200
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor.unsqueeze(1).to(device))
        loss = criterion(outputs, y_train_tensor.to(device))
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.unsqueeze(1).to(device))
            val_loss = criterion(val_outputs, y_val_tensor.to(device)).item()
        
        scheduler.step(val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
        
        # 早停检查
        if early_stopping(val_loss):
            print(f'Early stopping at epoch {epoch}')
            break
            
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'tuned_sensor_transformer.pth')
    
    print(f'最佳验证损失: {best_val_loss:.4f}')
    return model