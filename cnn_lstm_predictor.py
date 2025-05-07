import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 修改位置


# ====================== 基础设置 ======================
# 设置随机种子，确保结果可复现
torch.manual_seed(50)
np.random.seed(50)

# 强制使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("警告：GPU不可用，将使用CPU进行训练")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型参数设置
SEQUENCE_LENGTH = 10    # 序列长度，即用多少天的数据预测下一天，原值是10
EPOCHS = 100           # 训练轮次，原值是200
BATCH_SIZE = 16        # 批次大小，原值是32
LEARNING_RATE = 0.00005  # 学习率，原值是0.001
HIDDEN_SIZE = 32       # LSTM隐藏层大小，原值是64
NUM_LAYERS = 2         # LSTM层数，原值是2

# ====================== 数据处理类 ======================
class DataProcessor:
    """
    数据预处理类
    功能：加载数据、数据标准化、创建序列、划分数据集
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        # 为每个指标创建单独的标准化器
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
        
        # 划分训练集和测试集 (70% 训练, 30% 测试)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=50  #原值是42
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

# ====================== CNN-LSTM模型定义 ======================
class AirQualityCNNLSTM(nn.Module):
    """
    空气质量预测CNN-LSTM混合模型
    CNN用于提取空间特征，LSTM用于捕捉时间依赖关系
    """
    def __init__(self, sequence_length, n_features=7):
        super(AirQualityCNNLSTM, self).__init__()
        
        # CNN部分 - 提取空间特征
        self.cnn = nn.Sequential(
            # 第一个卷积层
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),  # 修改：原为32
            nn.BatchNorm1d(32),  #原32
            nn.ReLU(),


            # 第二个卷积层
            nn.Conv1d(32, 64, kernel_size=3, padding=1),   #原值：32，64，3，1
            nn.BatchNorm1d(64), #原值64
            nn.ReLU(),

        )
        
        # LSTM部分 - 处理时序关系
        self.lstm = nn.LSTM(
            input_size=64,  # 修改：与CNN输出通道匹配（原为64）
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.3  #原值0.2
        )
        
        # 全连接层 - 输出预测
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_features)
        )
    
    def forward(self, x):
        # CNN特征提取
        # 调整维度顺序：[batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        
        # 准备LSTM输入
        # [batch, channels, seq_len] -> [batch, seq_len, channels]
        lstm_in = cnn_out.permute(0, 2, 1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(lstm_in)
        
        # 只使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层预测
        output = self.fc(last_output)
        return output

# ====================== 可视化函数 ======================
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
    model = AirQualityCNNLSTM(SEQUENCE_LENGTH).to(device)
    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #这是原来用的adam
    # 新修改：使用AdamW优化器并添加权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # 修改：原为Adam
    
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
    data_file = r"C:\Users\lms\Desktop\2015-2025.xlsx" # 请替换为实际的数据文件路径
    main(data_file)