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
SEQUENCE_LENGTH = 10  # 序列长度，即用多少天的数据预测下一天，原本为10
EPOCHS = 100         # 训练轮次
BATCH_SIZE = 32      # 批次大小
LEARNING_RATE = 0.001  # 学习率

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
        # 获取所有污染物指标
        features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
        
        # 标准化每个指标的数据
        scaled_data = np.zeros((len(df), len(features)))
        for i, feature in enumerate(features):
            scaled_data[:, i] = self.scalers[feature].fit_transform(
                df[feature].values.reshape(-1, 1)
            ).ravel()
        
        # 创建序列
        X, y = self.create_sequences(scaled_data)
        
        # 划分训练集和测试集 (70% 训练, 30% 测试)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42   # 将原0.3改为0.2，即测试集占20%   # 保持随机种子确保可复现性
        )
        
        # 转换为PyTorch张量
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

# ====================== CNN模型定义 ======================
class AirQualityCNN(nn.Module):
    """
    空气质量预测CNN模型
    输入：[batch_size, sequence_length, n_features]
    输出：[batch_size, n_features]
    """
    def __init__(self, sequence_length, n_features=7):
        super(AirQualityCNN, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 计算全连接层的输入维度
        self.flatten_size = 64 * (sequence_length // 4)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_features)
        )
    
    def forward(self, x):
        # 调整输入维度顺序：[batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# ====================== 可视化函数 ======================
class Visualizer:
    """可视化工具类"""
    def __init__(self):
        self.features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
    
    def plot_training_loss(self, train_losses, val_losses):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
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

# ====================== 训练函数 ======================
def train_model(model, train_loader, test_loader, criterion, optimizer, device):
    """模型训练函数"""
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
            val_loss /= len(test_loader)
            val_losses.append(val_loss)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# ====================== 评估函数 ======================
def evaluate_model(model, test_loader, device):
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
    
    # 计算每个指标的MSE和R2分数
    features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
    for i, feature in enumerate(features):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)  # 修改位置
        print(f'{feature}:')
        print(f'  RMSE: {rmse:.4f}')  # 修改位置
        print(f'  MSE: {mse:.4f}')
        print(f'  MAE: {mae:.4f}')
        print(f'  R2 Score: {r2:.4f}')
    
    return y_true, y_pred

# ====================== 主函数 ======================
def main(data_file):
    """主函数"""
    # 初始化数据处理器
    data_processor = DataProcessor(sequence_length=SEQUENCE_LENGTH)
    
    # 加载并处理数据
    df = data_processor.load_data(data_file)
    train_loader, test_loader = data_processor.process_data(df)
    
    # 初始化模型
    model = AirQualityCNN(SEQUENCE_LENGTH).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    print("开始训练模型...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, device
    )
    
    # 评估模型
    print("\n模型评估结果:")
    y_true, y_pred = evaluate_model(model, test_loader, device)
    
    # 可视化结果
    visualizer = Visualizer()
    visualizer.plot_training_loss(train_losses, val_losses)
    visualizer.plot_predictions(y_true, y_pred)

if __name__ == "__main__":
    # 设置数据文件路径
    data_file =r"C:\Users\lms\Desktop\2015-2025.xlsx"  # 请替换为实际的数据文件路径
    main(data_file)