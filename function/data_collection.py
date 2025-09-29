import pandas as pd
import numpy as np
import time
from datetime import datetime

class SensorDataCollector:
    def __init__(self, csv_file='sensor_data.csv'):
        self.csv_file = csv_file
        self.data_buffer = []
        
    def collect_data_point(self, t1, h1, t2, t, h2_measured, h2_actual=None):
        """
        收集单个数据点
        t1: 初始温度
        h1: 初始湿度
        t2: 目标温度  
        t: 响应时间
        h2_measured: 传感器在时间t测得的湿度
        h2_actual: 实际环境湿度（如果已知）
        """
        data_point = {
            'timestamp': datetime.now().isoformat(),
            't1': t1,
            'h1': h1,
            't2': t2,
            't': t,
            'h2_measured': h2_measured,
            'h2_actual': h2_actual,
            'delta_t': t2 - t1,  # 温度差
            'tau_estimate': self.estimate_tau(h1, h2_measured, t)  # 估算的时间常数
        }
        self.data_buffer.append(data_point)
        
    def estimate_tau(self, h1, h2_measured, t):
        """估算响应时间常数"""
        if abs(h2_measured - h1) < 0.1:  # 避免除零
            return float('inf')
        # 简单的一阶系统反推时间常数
        ratio = (h2_measured - h1) / (100 - h1)  # 假设最终值为100
        if ratio <= 0 or ratio >= 1:
            return float('inf')
        return -t / np.log(1 - ratio)
    
    def save_data(self):
        """保存数据到CSV文件"""
        df = pd.DataFrame(self.data_buffer)
        df.to_csv(self.csv_file, index=False)
        print(f"已保存 {len(self.data_buffer)} 条数据到 {self.csv_file}")
        
    def load_data(self):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(self.csv_file)
            return df
        except FileNotFoundError:
            print(f"文件 {self.csv_file} 不存在")
            return pd.DataFrame()

# 模拟数据收集过程
def simulate_data_collection():
    collector = SensorDataCollector('real_sensor_data.csv')
    
    # 模拟不同环境条件下的数据收集
    environments = [
        {'t1': 20, 't2': 30, 'h1': 40},
        {'t1': 25, 't2': 15, 'h1': 60},
        {'t1': 30, 't2': 35, 'h1': 50},
        {'t1': 15, 't2': 25, 'h1': 30},
    ]
    
    for env in environments:
        t1, t2, h1 = env['t1'], env['t2'], env['h1']
        
        # 模拟传感器响应过程
        real_h2 = 30 + (t2 - 10) * 1.5  # 真实环境湿度
        tau = 20 + abs(t2 - t1) * 2     # 响应时间常数与温差相关
        
        # 在不同时间点收集数据
        time_points = [0, 5, 10, 15, 20, 30, 45, 60]
        for t in time_points:
            # 传感器响应模型
            h2_measured = h1 + (real_h2 - h1) * (1 - np.exp(-t/tau)) + np.random.normal(0, 1)
            collector.collect_data_point(t1, h1, t2, t, h2_measured, real_h2)
    
    collector.save_data()
    return collector

# 运行模拟数据收集
collector = simulate_data_collection()