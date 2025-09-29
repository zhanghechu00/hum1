import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math

class SensorTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(SensorTransformer, self).__init__()
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 预测h2
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 取最后一个时间步的输出进行预测
        output = self.output_layer(x[:, -1, :])  # (batch_size, 1)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 数据集类
class SensorDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 数据生成函数（模拟物理过程）
def generate_sensor_data(num_samples=10000):
    """
    生成模拟传感器数据
    基于简单的物理模型：湿度传感器响应遵循一阶指数衰减
    """
    data = []
    targets = []
    
    for _ in range(num_samples):
        # 随机生成初始条件
        t1 = np.random.uniform(10, 40)  # 初始温度 10-40°C
        h1 = np.random.uniform(20, 80)  # 初始湿度 20-80%RH
        t2 = np.random.uniform(10, 40)  # 目标温度 10-40°C
        
        # 随机响应时间常数（基于温度影响）
        tau = np.random.uniform(5, 60)  # 响应时间常数 5-60秒
        
        # 随机观测时间
        t = np.random.uniform(0, tau * 3)  # 观测时间 0-3*tau
        
        # 真实的目标湿度（假设与温度相关）
        h2_real = 30 + (t2 - 10) * 1.5 + np.random.normal(0, 2)  # 简化的关系
        h2_real = np.clip(h2_real, 20, 90)
        
        # 传感器在时间t的实际读数（用于训练）
        h_sensor_t = h1 + (h2_real - h1) * (1 - np.exp(-t/tau))
        
        # 输入特征：[t1, h1, t2, t]
        input_features = [t1, h1, t2, t]
        data.append(input_features)
        
        # 目标：真实环境湿度h2
        targets.append([h2_real])
    
    return np.array(data), np.array(targets)

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x.unsqueeze(1))  # 添加序列维度
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x.unsqueeze(1))
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_sensor_transformer.pth')
    
    print(f'Best validation loss: {best_loss:.4f}')

# 预测函数
def predict_humidity(model, t1, h1, t2, t):
    """
    预测环境湿度h2
    
    参数:
    - model: 训练好的模型
    - t1: 初始温度
    - h1: 初始湿度
    - t2: 新环境温度
    - t: 进入新环境的时间
    
    返回:
    - 预测的环境湿度h2
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 准备输入数据
    input_data = torch.tensor([[t1, h1, t2, t]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prediction = model(input_data.unsqueeze(1))  # 添加序列维度
        return prediction.item()

# 主程序
def main():
    # 生成训练数据
    print("生成训练数据...")
    X, y = generate_sensor_data(50000)
    
    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 分割训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_dataset = SensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                 torch.tensor(y_train, dtype=torch.float32))
    val_dataset = SensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                               torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 创建模型
    model = SensorTransformer(input_dim=4, d_model=64, nhead=4, num_layers=3)
    
    # 训练模型
    print("开始训练...")
    train_model(model, train_loader, val_loader, num_epochs=100)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_sensor_transformer.pth'))
    
    # 测试预测
    print("\n测试预测:")
    test_cases = [
        (25, 50, 30, 10),  # t1=25°C, h1=50%RH, t2=30°C, t=10s
        (20, 30, 35, 20),  # t1=20°C, h1=30%RH, t2=35°C, t=20s
        (30, 70, 25, 5),   # t1=30°C, h1=70%RH, t2=25°C, t=5s
    ]
    
    for t1, h1, t2, t in test_cases:
        predicted_h2 = predict_humidity(model, t1, h1, t2, t)
        print(f"输入: t1={t1}°C, h1={h1}%RH, t2={t2}°C, t={t}s")
        print(f"预测的h2: {predicted_h2:.2f}%RH\n")

if __name__ == "__main__":
    main()