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
NUM_CHANNELS = [32, 64, 128]  # TCN的通道数，3层结构
KERNEL_SIZE = 3        # 卷积核大小
DROPOUT = 0.2         # Dropout比率

# ====================== 数据处理类 ======================
class DataProcessor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        # 同时使用标准化和归一化
        self.scalers = {
            'AQI': MinMaxScaler(feature_range=(0, 1)),
            'PM2.5': MinMaxScaler(feature_range=(0, 1)),
            'PM10': MinMaxScaler(feature_range=(0, 1)),
            'CO': MinMaxScaler(feature_range=(0, 1)),
            'SO2': MinMaxScaler(feature_range=(0, 1)),
            'NO2': MinMaxScaler(feature_range=(0, 1)),
            'O3_8h': MinMaxScaler(feature_range=(0, 1))
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
    
    def preprocess_data(self, df):
        """数据预处理：清洗、标准化、归一化"""

        
        return df
    
    def process_data(self, df):
        """完整的数据预处理流程"""
        # 1. 基础预处理
        df = self.preprocess_data(df)
        features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
        
        # 2. 数据标准化和归一化
        scaled_data = np.zeros((len(df), len(features)))
        for i, feature in enumerate(features):
            scaled_data[:, i] = self.scalers[feature].fit_transform(
                df[feature].values.reshape(-1, 1)
            ).ravel()
        
        # 3. 创建序列
        X, y = self.create_sequences(scaled_data)
        print(f"创建的序列数据形状: X={X.shape}, y={y.shape}")
        
        # 4. 划分数据集 (90% 训练, 10% 测试)
        split_idx = int(len(X) * 0.9)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 5. 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
        
        # 6. 调整维度顺序为TCN要求的格式：[batch, channels, length]
        X_train = X_train.permute(0, 2, 1)
        X_test = X_test.permute(0, 2, 1)
        
        # 7. 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        return train_loader, test_loader

# ====================== TCN基础模块定义 ======================
class Chomp1d(nn.Module):
    """TCN中用于因果卷积的裁剪层"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """TCN的基本模块，包含因果卷积、归一化、激活和残差连接"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """TCN主体网络，由多个TemporalBlock组成"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ====================== TCN模型定义 ======================
class AirQualityTCN(nn.Module):
    def __init__(self, input_size=7, output_size=7, num_channels=NUM_CHANNELS, 
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super(AirQualityTCN, self).__init__()
        
        # TCN主体
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # TCN特征提取
        tcn_out = self.tcn(x)
        # 取最后一个时间步的输出
        out = self.fc(tcn_out[:, :, -1])
        return out

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
def train_model(model, train_loader, test_loader, criterion, optimizer, device):
    """模型训练函数"""
    train_losses = []
    val_losses = []
    
    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
    
    # 保存最终模型
    torch.save(model.state_dict(), 'tcn_model.pth')
    
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
    model = AirQualityTCN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, device
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
    torch.save(model.state_dict(), 'air_quality_tcn_model.pth')
    print("\n模型已保存为: air_quality_tcn_model.pth")

if __name__ == '__main__':
    # 设置数据文件路径
    data_file = r"C:\Users\lms\Desktop\2015-2025.xlsx" # 替换为实际的数据文件路径
    main(data_file)