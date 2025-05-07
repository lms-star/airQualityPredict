import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# ====================== 基础设置 ======================
# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 强制使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("警告：GPU不可用，将使用CPU进行训练")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型参数设置
SEQUENCE_LENGTH = 10    # 序列长度，即用多少天的数据预测下一天
EPOCHS = 100           # 训练轮次
BATCH_SIZE = 32        # 批次大小
LEARNING_RATE = 0.001  # 学习率
HIDDEN_SIZE = 128      # LSTM隐藏层大小
NUM_LAYERS = 2         # LSTM层数

# ====================== 数据处理类 ======================
class DataProcessor:
    """
    数据预处理类
    功能：加载数据、数据标准化、创建序列、划分数据集
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scalers = {
            'AQI': MinMaxScaler(),
            'PM2.5': MinMaxScaler(),
            'PM10': MinMaxScaler(),
            'CO': MinMaxScaler(),
            'SO2': MinMaxScaler(),
            'NO2': MinMaxScaler(),
            'O3_8h': MinMaxScaler()
        }
    
    def load_data(self, file_path):
        """读取数据文件（支持Excel和CSV格式）"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        print("数据加载完成，形状:", df.shape)
        return df
    
    def create_sequences(self, data):
        """创建时间序列数据"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def process_data(self, df):
        """数据预处理主函数"""
        features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
        
        # 标准化每个指标的数据
        scaled_data = np.zeros((len(df), len(features)))
        for i, feature in enumerate(features):
            scaled_data[:, i] = self.scalers[feature].fit_transform(
                df[feature].values.reshape(-1, 1)
            ).ravel()
        
        # 创建序列
        X, y = self.create_sequences(scaled_data)
        print("序列数据形状:", X.shape, y.shape)
        
        # 划分训练集和测试集 (90% 训练, 10% 测试)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        # 转换为PyTorch张量并移至GPU
        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        return train_loader, test_loader

# ====================== LSTM模型定义 ======================
class AirQualityLSTM(nn.Module):
    """
    空气质量预测LSTM模型
    输入：[batch_size, sequence_length, n_features]
    输出：[batch_size, n_features]
    """
    def __init__(self, input_size=7, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(AirQualityLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, input_size)
        )
    
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 只使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        return output

# ====================== 可视化类 ======================
class Visualizer:
    """可视化工具类"""
    def __init__(self):
        self.features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
    
    def plot_training_loss(self, train_losses, val_losses):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', linewidth=2)
        plt.plot(val_losses, label='验证损失', linewidth=2)
        plt.title('模型训练过程中的损失变化')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_predictions(self, y_true, y_pred):
        """绘制每个污染物的预测结果对比图"""
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        axes = axes.ravel()
        
        for idx, (feature, ax) in enumerate(zip(self.features, axes)):
            ax.plot(y_true[:, idx], label='真实值', marker='o')
            ax.plot(y_pred[:, idx], label='预测值', marker='x')
            ax.set_title(f'{feature} 预测结果对比')
            ax.set_xlabel('样本')
            ax.set_ylabel('浓度')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics(self, metrics):
        """绘制评估指标可视化图"""
        metrics_names = ['RMSE', 'R2', 'MSE', 'MAE']
        features = self.features
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (metric_name, ax) in enumerate(zip(metrics_names, axes)):
            metric_values = [metrics[feature][metric_name] for feature in features]
            
            ax.bar(features, metric_values)
            ax.set_title(f'{metric_name} by Feature')
            ax.set_xlabel('Features')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# ====================== 训练函数 ======================
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    """模型训练函数"""
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # 记录损失
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(test_loader))
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

# ====================== 评估函数 ======================
def evaluate_model(model, test_loader):
    """模型评估函数"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # 合并所有批次的预测结果
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_targets)
    
    # 计算每个指标的评估指标
    metrics = {}
    features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
    
    for i, feature in enumerate(features):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        metrics[feature] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f'\n{feature}:')
        print(f'  MSE: {mse:.4f}')
        print(f'  RMSE: {rmse:.4f}')
        print(f'  MAE: {mae:.4f}')
        print(f'  R2 Score: {r2:.4f}')
    
    return y_true, y_pred, metrics

# ====================== 主函数 ======================
def main(data_path):
    """主函数"""
    # 初始化数据处理器
    data_processor = DataProcessor(sequence_length=SEQUENCE_LENGTH)
    
    # 加载和处理数据
    df = data_processor.load_data(data_path)
    train_loader, test_loader = data_processor.process_data(df)
    
    # 初始化模型
    model = AirQualityLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, EPOCHS
    )
    
    # 评估模型
    print("\n模型评估:")
    y_true, y_pred, metrics = evaluate_model(model, test_loader)
    
    # 可视化结果
    visualizer = Visualizer()
    visualizer.plot_training_loss(train_losses, val_losses)
    visualizer.plot_predictions(y_true, y_pred)
    visualizer.plot_metrics(metrics)
    
    # 保存模型
    torch.save(model.state_dict(), 'air_quality_lstm_model.pth')
    print("\n模型已保存为: air_quality_lstm_model.pth")

if __name__ == '__main__':
    # 设置数据文件路径
    data_file = r"C:\Users\lms\Desktop\2015-2025.xlsx" # 替换为实际的数据文件路径
    main(data_file)