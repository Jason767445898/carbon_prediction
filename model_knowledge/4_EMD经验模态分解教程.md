# EMD经验模态分解在金融时间序列预处理中的应用

## 1. EMD原理详解

### 1.1 什么是EMD？

经验模态分解（Empirical Mode Decomposition, EMD）是一种自适应的时间序列分解方法，能够：

- **自适应分解**：不需要预设基函数，完全数据驱动
- **处理非线性非平稳信号**：适合金融时间序列的复杂特性
- **多尺度分析**：将信号分解为不同频率的本征模态函数（IMF）
- **保持时频局部化特性**：每个IMF都保持原始信号的时间特性

### 1.2 EMD数学原理

EMD将信号分解为多个本征模态函数（IMF）和一个趋势项：

```
x(t) = Σc_i(t) + r(t)
```

其中：
- `c_i(t)` 是第i个IMF分量
- `r(t)` 是残余分量（趋势）

IMF必须满足两个条件：
1. 极值点数量与零点数量相差不超过1
2. 任意点处，由极大值连线的上包络和极小值连线的下包络的均值为零

### 1.3 在金融中的应用价值

- **市场趋势分解**：分离长期趋势和短期波动
- **噪声过滤**：去除高频噪声，保留有效信号
- **周期识别**：发现不同时间尺度的市场周期
- **异常检测**：识别市场异常波动

## 2. EMD在金融数据中的实现

### 2.1 环境准备

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.interpolate import interp1d
import yfinance as yf
from PyEMD import EMD, EEMD, CEEMDAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("EMD相关库导入成功")
```

### 2.2 获取和准备金融数据

```python
def get_financial_data(symbol='AAPL', start_date='2022-01-01', end_date='2023-12-31'):
    """获取金融数据"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except:
        # 创建模拟金融数据
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        t = np.arange(len(dates))
        
        # 长期趋势
        trend = 100 + 0.05 * t + 0.0001 * t**2
        
        # 季节性成分
        seasonal_annual = 5 * np.sin(2 * np.pi * t / 365)
        seasonal_monthly = 2 * np.sin(2 * np.pi * t / 30)
        seasonal_weekly = 1 * np.sin(2 * np.pi * t / 7)
        
        # 随机噪声
        noise = np.random.normal(0, 1, len(dates))
        
        # 合成价格
        prices = trend + seasonal_annual + seasonal_monthly + seasonal_weekly + noise
        prices = np.maximum(prices, 50)
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(prices))),
            'High': prices * (1 + np.random.uniform(0, 0.02, len(prices))),
            'Low': prices * (1 - np.random.uniform(0, 0.02, len(prices))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        return data

# 获取数据
print("获取金融数据...")
stock_data = get_financial_data('AAPL', '2022-01-01', '2023-12-31')
prices = stock_data['Close'].values
dates = stock_data.index

print(f"数据长度: {len(prices)}")
print(f"价格范围: {prices.min():.2f} - {prices.max():.2f}")
```

### 2.3 EMD分解实现

```python
def perform_emd_analysis(data, method='EMD'):
    """使用PyEMD进行EMD分析"""
    
    if method == 'EMD':
        emd = EMD()
        emd.emd(data)
        imfs, residue = emd.get_imfs_and_residue()
    elif method == 'EEMD':
        # 集成经验模态分解 - 添加噪声提高鲁棒性
        eemd = EEMD()
        eemd.noise_seed(42)
        eimfs = eemd.eemd(data, max_imf=8)
        imfs = eimfs[:-1]
        residue = eimfs[-1]
    elif method == 'CEEMDAN':
        # 完整集成经验模态分解
        ceemdan = CEEMDAN()
        ceemdan.noise_seed(42)
        cimfs = ceemdan.ceemdan(data, max_imf=8)
        imfs = cimfs[:-1]
        residue = cimfs[-1]
    
    return imfs, residue

def plot_emd_results(original_data, imfs, residue, dates, title="EMD分解结果"):
    """可视化EMD分解结果"""
    
    n_imfs = len(imfs)
    fig, axes = plt.subplots(n_imfs + 2, 1, figsize=(15, 3 * (n_imfs + 2)))
    
    # 原始信号
    axes[0].plot(dates, original_data, 'b-', linewidth=1)
    axes[0].set_title('原始价格序列', fontsize=12)
    axes[0].set_ylabel('价格')
    axes[0].grid(True, alpha=0.3)
    
    # 各个IMF分量
    for i, imf in enumerate(imfs):
        axes[i+1].plot(dates, imf, linewidth=1)
        axes[i+1].set_title(f'IMF {i+1}', fontsize=12)
        axes[i+1].set_ylabel('振幅')
        axes[i+1].grid(True, alpha=0.3)
    
    # 残余分量
    axes[-1].plot(dates, residue, 'r-', linewidth=2)
    axes[-1].set_title('残余分量（趋势）', fontsize=12)
    axes[-1].set_ylabel('价格')
    axes[-1].set_xlabel('时间')
    axes[-1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 执行EMD分解
print("执行EMD分解...")
imfs_emd, residue_emd = perform_emd_analysis(prices, method='EMD')
print(f"EMD分解得到 {len(imfs_emd)} 个IMF分量")

# 可视化结果
plot_emd_results(prices, imfs_emd, residue_emd, dates, "标准EMD分解结果")
```

### 2.4 IMF分量分析

```python
def analyze_imf_properties(imfs, residue, original_data):
    """分析IMF分量的特性"""
    
    analysis_results = []
    
    for i, imf in enumerate(imfs):
        # 计算统计特性
        energy = np.sum(imf**2)
        
        # 计算主频率
        freqs = np.fft.fftfreq(len(imf), 1)
        fft_vals = np.abs(np.fft.fft(imf))
        dominant_freq_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        # 计算周期
        period = 1 / abs(dominant_freq) if dominant_freq != 0 else np.inf
        
        analysis_results.append({
            'IMF': f'IMF{i+1}',
            'Mean': np.mean(imf),
            'Std': np.std(imf),
            'Energy_Ratio': energy / np.sum(original_data**2),
            'Period_Days': period,
            'Max_Amplitude': np.max(np.abs(imf))
        })
    
    # 添加残余分量
    residue_energy = np.sum(residue**2)
    analysis_results.append({
        'IMF': 'Residue',
        'Mean': np.mean(residue),
        'Std': np.std(residue),
        'Energy_Ratio': residue_energy / np.sum(original_data**2),
        'Period_Days': np.inf,
        'Max_Amplitude': np.max(np.abs(residue))
    })
    
    return pd.DataFrame(analysis_results)

# 分析IMF特性
imf_analysis = analyze_imf_properties(imfs_emd, residue_emd, prices)
print("EMD分量特性分析:")
print(imf_analysis.round(4))

# 可视化能量分布和周期
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(imf_analysis['IMF'], imf_analysis['Energy_Ratio'])
plt.title('各分量能量占比')
plt.xlabel('分量')
plt.ylabel('能量占比')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
periods = imf_analysis['Period_Days'][:-1]  # 排除残余
periods = [p if p != np.inf else 1000 for p in periods]
plt.semilogy(range(1, len(periods)+1), periods, 'o-')
plt.title('各IMF分量周期')
plt.xlabel('IMF序号')
plt.ylabel('周期（天，对数尺度）')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.bar(imf_analysis['IMF'], imf_analysis['Max_Amplitude'])
plt.title('各分量最大振幅')
plt.xlabel('分量')
plt.ylabel('最大振幅')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### 2.5 EMD去噪应用

```python
def emd_denoising(data, noise_threshold=0.01):
    """使用EMD进行去噪"""
    
    # 执行EMD分解
    emd = EMD()
    imfs = emd.emd(data)
    
    # 自动识别噪声分量
    noise_imfs = []
    for i, imf in enumerate(imfs[:-1]):  # 排除残余
        energy_ratio = np.sum(imf**2) / np.sum(data**2)
        if energy_ratio < noise_threshold:
            noise_imfs.append(i)
    
    print(f"识别的噪声IMF分量: {noise_imfs}")
    
    # 重构去噪后的信号
    denoised_signal = np.sum([imfs[i] for i in range(len(imfs)) if i not in noise_imfs], axis=0)
    
    return denoised_signal, imfs, noise_imfs

def compare_denoising_methods(original_data):
    """比较不同去噪方法"""
    
    # 1. EMD去噪
    denoised_emd, _, _ = emd_denoising(original_data)
    
    # 2. 移动平均去噪
    denoised_ma = pd.Series(original_data).rolling(window=5, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    # 3. 高斯滤波去噪
    from scipy.ndimage import gaussian_filter1d
    denoised_gaussian = gaussian_filter1d(original_data, sigma=1.0)
    
    # 可视化比较
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(original_data, 'b-', alpha=0.7, label='原始数据')
    plt.title('原始数据')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(original_data, 'b-', alpha=0.5, label='原始数据')
    plt.plot(denoised_emd, 'r-', linewidth=2, label='EMD去噪')
    plt.title('EMD去噪结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(original_data, 'b-', alpha=0.5, label='原始数据')
    plt.plot(denoised_ma, 'g-', linewidth=2, label='移动平均去噪')
    plt.title('移动平均去噪结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(original_data, 'b-', alpha=0.5, label='原始数据')
    plt.plot(denoised_gaussian, 'm-', linewidth=2, label='高斯滤波去噪')
    plt.title('高斯滤波去噪结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return denoised_emd, denoised_ma, denoised_gaussian

# 执行去噪比较
denoised_signals = compare_denoising_methods(prices)
```

## 3. EMD在金融预测中的应用

### 3.1 EMD特征工程

```python
def create_emd_features(data, window_size=60):
    """基于EMD创建预测特征"""
    
    features_list = []
    
    for i in range(window_size, len(data)):
        window_data = data[i-window_size:i]
        
        # EMD分解
        emd = EMD()
        imfs = emd.emd(window_data, max_imf=6)
        
        features = {}
        
        # 为每个IMF创建特征
        for j, imf in enumerate(imfs[:-1]):  # 排除残余
            features[f'IMF{j+1}_energy'] = np.sum(imf**2)
            features[f'IMF{j+1}_std'] = np.std(imf)
            features[f'IMF{j+1}_mean'] = np.mean(imf)
            if len(imf) > 1:
                features[f'IMF{j+1}_trend'] = (imf[-1] - imf[0]) / len(imf)
        
        # 残余分量特征
        residue = imfs[-1]
        features['residue_trend'] = (residue[-1] - residue[0]) / len(residue) if len(residue) > 1 else 0
        features['residue_mean'] = np.mean(residue)
        
        # 频率域特征
        high_freq = np.sum(imfs[:2], axis=0) if len(imfs) >= 2 else np.zeros_like(window_data)
        mid_freq = np.sum(imfs[2:4], axis=0) if len(imfs) >= 4 else np.zeros_like(window_data)
        low_freq = np.sum(imfs[4:], axis=0) if len(imfs) > 4 else residue
        
        features['high_freq_energy'] = np.sum(high_freq**2)
        features['mid_freq_energy'] = np.sum(mid_freq**2)
        features['low_freq_energy'] = np.sum(low_freq**2)
        
        features['timestamp'] = i
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# 创建EMD特征
print("创建EMD特征...")
emd_features_df = create_emd_features(prices, window_size=60)
print(f"EMD特征数据框形状: {emd_features_df.shape}")
```

### 3.2 EMD预测模型

```python
def build_emd_prediction_model(price_data, emd_features, prediction_horizon=5):
    """构建基于EMD特征的预测模型"""
    
    # 创建目标变量（未来收益率）
    targets = []
    valid_indices = []
    
    for i, row in emd_features.iterrows():
        timestamp = int(row['timestamp'])
        if timestamp + prediction_horizon < len(price_data):
            current_price = price_data[timestamp]
            future_price = price_data[timestamp + prediction_horizon]
            future_return = (future_price - current_price) / current_price
            targets.append(future_return)
            valid_indices.append(i)
    
    # 准备特征和目标
    X = emd_features.loc[valid_indices].drop('timestamp', axis=1)
    y = np.array(targets)
    
    print(f"预测数据形状: X={X.shape}, y={y.shape}")
    
    # 分割数据
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 评估
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"模型性能:")
    print(f"训练集 - MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
    print(f"测试集 - MSE: {test_mse:.6f}, R²: {test_r2:.4f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n特征重要性排序:")
    print(feature_importance.head(10))
    
    return model, scaler, feature_importance

# 构建预测模型
model, scaler, feature_importance = build_emd_prediction_model(prices, emd_features_df, prediction_horizon=5)

# 可视化特征重要性
plt.figure(figsize=(12, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('特征重要性')
plt.title('EMD特征重要性排序')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

## 4. EMD实际应用建议

### 4.1 EMD vs 传统方法比较

| 特性 | EMD | 傅里叶变换 | 小波变换 |
|------|-----|-----------|----------|
| **自适应性** | 高 | 低 | 中 |
| **非平稳信号处理** | 优秀 | 差 | 好 |
| **计算复杂度** | 高 | 低 | 中 |
| **频率分辨率** | 自适应 | 固定 | 多分辨率 |
| **时间局部化** | 优秀 | 差 | 好 |

### 4.2 实践建议

1. **数据预处理**：EMD对边界效应敏感，建议使用镜像延拓等方法
2. **参数选择**：根据数据特性选择合适的停止准则和IMF数量
3. **噪声处理**：使用EEMD或CEEMDAN提高噪声鲁棒性
4. **特征选择**：重点关注能量和趋势特征，它们通常最有预测价值
5. **模型验证**：使用时间序列交叉验证避免数据泄露

### 4.3 应用场景

- **趋势预测**：使用低频IMF和残余分量
- **波动率建模**：使用高频IMF分量
- **异常检测**：监控IMF分量的异常变化
- **风险管理**：分解不同时间尺度的风险因子