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
torch.manual_seed(50)
np.random.seed(50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 修改后的参数设置
SEQUENCE_LENGTH = 10  # 序列长度
EPOCHS = 50  # 训练轮次
BATCH_SIZE = 16  # 批次大小
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 32
NUM_LAYERS = 1


# ====================== 注意力机制模块 ======================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(attention_weights * x, dim=1)
        return context, attention_weights


# ====================== CNN-LSTM-Attention模型定义 ======================
class PM25Predictor(nn.Module):  # 修改模型名称
    def __init__(self, sequence_length, n_features=1):  # 修改输入特征数为1
        super(PM25Predictor, self).__init__()

        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.attention = AttentionLayer(HIDDEN_SIZE * 2)

        # 修改输出层为1个神经元
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # 只输出PM2.5
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        lstm_in = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        context, attention_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attention_weights  # 返回单个预测值


# ====================== 数据处理类 ======================
class DataProcessor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()  # 仅保留PM2.5的归一化器



    def load_data(self, file_path):
        """读取数据文件"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df['PM2.5'].values

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])  # 只取PM2.5值
        return np.array(X), np.array(y)

    # def process_data(self, df):
    #     # 仅处理PM2.5特征
    #     pm25_data = df.reshape(-1, 1)
    #     scaled_data = self.scaler.fit_transform(pm25_data)
    #
    #     # 创建序列
    #     X, y = self.create_sequences(scaled_data.flatten())
    #     print("序列数据形状:", X.shape, y.shape)
    #
    #     # 调整输入数据的维度
    #     X = X.reshape(-1, self.sequence_length, 1)
    #     y = y.reshape(-1, 1)

    def process_data(self, df):
        # 划分训练测试集（时间序列必须按顺序）
        train_size = int(len(df) * 0.9)
        train_data = df[:train_size]
        test_data = df[train_size:]  # 移除了序列衔接的处理方式

        # 训练集归一化
        self.scaler.fit(train_data.reshape(-1, 1))
        scaled_train = self.scaler.transform(train_data.reshape(-1, 1))
        scaled_test = self.scaler.transform(test_data.reshape(-1, 1))

        # 生成序列
        X_train, y_train = self.create_sequences(scaled_train.flatten())
        X_test, y_test = self.create_sequences(scaled_test.flatten())

        # 调整维度
        X_train = X_train.reshape(-1, self.sequence_length, 1)
        X_test = X_test.reshape(-1, self.sequence_length, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # 转换为张量
        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)

        return train_loader, test_loader


# ====================== 可视化函数 ======================
class Visualizer:
    def plot_training_loss(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.title('训练过程损失变化')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_single_prediction(self, y_true, y_pred):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='真实值', alpha=0.7)
        plt.plot(y_pred, label='预测值', linestyle='--')
        plt.title('PM2.5浓度预测对比')
        plt.xlabel('样本')
        plt.ylabel('浓度')
        plt.legend()
        plt.grid(True)
        plt.show()


# ====================== 训练函数 ======================
def train_model(model, train_loader, test_loader, criterion, optimizer, device):
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
            val_loss /= len(test_loader)
            val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


# ====================== 评估函数 ======================
def evaluate_model(model, test_loader, scaler):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs, _ = model(batch_X)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    # 逆标准化
    y_true_orig = scaler.inverse_transform(y_true)
    y_pred_orig = scaler.inverse_transform(y_pred)

    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)

    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R2 Score: {r2:.4f}')

    return y_true_orig, y_pred_orig


# ====================== 主函数 ======================
def main(data_file):
    # 初始化组件
    data_processor = DataProcessor(sequence_length=SEQUENCE_LENGTH)
    visualizer = Visualizer()

    # 加载数据
    df = data_processor.load_data(data_file)
    train_loader, test_loader = data_processor.process_data(df)

    # 初始化模型
    model = PM25Predictor(SEQUENCE_LENGTH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss()

    # 训练
    print("\n开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, device
    )
    visualizer.plot_training_loss(train_losses, val_losses)

    # 评估
    print("\n评估结果：")
    y_true, y_pred = evaluate_model(model, test_loader, data_processor.scaler)
    visualizer.plot_single_prediction(y_true, y_pred)



if __name__ == "__main__":
    data_path = r"C:\Users\lms\Desktop\2015-2025.xlsx"
    try:
        main(data_path)
    except Exception as e:
        print(f"运行错误：{str(e)}")