#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
碳价格预测系统 - LSTM + Attention机制
使用 data.DTA 数据文件，采用 LSTM 和 Attention 机制的结合来预测碳价格
输出包括：Excel表格、TXT文档、图片等分析结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# SHAP分析
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available, skipping interpretability analysis")

warnings.filterwarnings('ignore')

# 设置英文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    'data_file': 'data.dta',
    'target_column': 'coal_price',  # 煤炭价格列
    'sequence_length': 60,  # 序列长度
    'test_size': 0.2,
    'validation_size': 0.1,
    'epochs': 300,  # 煤炭价格优化第一轮最优配置
    'batch_size': 32,  # 煤炭价格优化第一轮最优配置
    'learning_rate': 0.00015,  # 煤炭价格优化第一轮最优配置(Config3)
    'lstm_units': 384,  # 煤炭价格优化第一轮最优配置:增加至384
    'lstm_units_2': 192,  # 煤炭价格优化第一轮最优配置:增加至192
    'lstm_units_3': 96,  # 煤炭价格优化第一轮最优配置:增加至96
    'attention_dim': 192,  # 煤炭价格优化第一轮最优配置:增加至192
    'dropout_rate': 0.4,  # 煤炭价格优化第一轮最优配置:增加至0.4
    'l2_reg': 0.001,  # 煤炭价格优化第一轮最优配置:增加至0.001
    'gradient_clip': 1.0,  # 梯度裁剪阈值
}

# 输出目录
OUTPUT_DIR = 'outputs'
for sub_dir in ['logs', 'reports', 'visualizations']:
    os.makedirs(os.path.join(OUTPUT_DIR, sub_dir), exist_ok=True)

# ============================================================================
# 辅助函数
# ============================================================================
def create_attention_layer(input_tensor, attention_dim):
    """创建Keras兼容的Scaled Dot-Product Attention层
    
    参数:
        input_tensor: 输入张量 shape=(batch_size, sequence_length, features)
        attention_dim: attention的维度
    
    返回:
        context_vector: 上下文向量 shape=(batch_size, attention_dim)
    """
    # Query, Key, Value 转换
    query = layers.Dense(attention_dim, name='attention_query')(input_tensor)
    key = layers.Dense(attention_dim, name='attention_key')(input_tensor)
    value = layers.Dense(attention_dim, name='attention_value')(input_tensor)
    
    # 使用Keras layers进行矩阵乘法和缩放
    # scores = Q * K^T / sqrt(d_k)
    scores = layers.Dot(axes=[2, 2])([query, key])
    scores = layers.Lambda(lambda x: x / tf.math.sqrt(tf.cast(attention_dim, tf.float32)))(scores)
    
    # Softmax获得attention weights
    attention_weights = layers.Softmax(axis=-1, name='attention_weights')(scores)
    
    # 应用attention weights到value
    # context = attention_weights @ value
    context = layers.Dot(axes=[2, 1])([attention_weights, value])
    
    # 全局平均池化聚合到单个上下文向量
    context_vector = layers.GlobalAveragePooling1D(name='attention_pooling')(context)
    
    return context_vector

def build_lstm_attention_model(sequence_length, n_features, lstm_units, attention_dim):
    """
    构建深层LSTM + Attention组合模型（三层LSTM + 增强正则化）
    """
    inputs = layers.Input(shape=(sequence_length, n_features))
    
    # 第一层LSTM - 最大容量
    lstm_out = layers.LSTM(
        CONFIG['lstm_units'], 
        return_sequences=True,
        dropout=0.2,  # LSTM内部dropout
        recurrent_dropout=0.1,
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']),
        recurrent_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg'])
    )(inputs)
    lstm_out = layers.BatchNormalization()(lstm_out)
    
    # 第二层LSTM - 中等容量
    lstm_out = layers.LSTM(
        CONFIG['lstm_units_2'], 
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.1,
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']),
        recurrent_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg'])
    )(lstm_out)
    lstm_out = layers.BatchNormalization()(lstm_out)
    
    # 第三层LSTM - 较小容量
    lstm_out = layers.LSTM(
        CONFIG['lstm_units_3'], 
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.1,
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']),
        recurrent_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg'])
    )(lstm_out)
    lstm_out = layers.BatchNormalization()(lstm_out)
    
    # Attention层 + 残差连接
    attention_out = create_attention_layer(lstm_out, attention_dim)
    
    # 残差连接：将LSTM聚合表示与attention输出相加
    lstm_pooled = layers.GlobalAveragePooling1D(name='lstm_pooling')(lstm_out)
    
    # 维度匹配：如果attention_dim与lstm_units_3不同，需要投影
    if attention_dim != CONFIG['lstm_units_3']:
        lstm_pooled = layers.Dense(attention_dim, name='residual_projection')(lstm_pooled)
    
    # 残差连接：Add层
    combined = layers.Add(name='residual_connection')([lstm_pooled, attention_out])
    
    # Layer Normalization（Transformer标准组件）
    combined = layers.LayerNormalization(epsilon=1e-6, name='layer_norm')(combined)
    
    # 使用combined作为后续全连接层的输入
    attention_out = combined
    
    # 全连接层（Feed-Forward Network）
    dense = layers.Dense(128, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']))(attention_out)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(CONFIG['dropout_rate'])(dense)
    
    dense = layers.Dense(64, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']))(dense)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(CONFIG['dropout_rate'])(dense)
    
    dense = layers.Dense(32, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['l2_reg']))(dense)
    dense = layers.Dropout(CONFIG['dropout_rate'] / 2)(dense)
    
    # 输出层
    outputs = layers.Dense(1)(dense)
    
    # 自定义方向感知损失函数（修复NaN问题）
    def directional_loss(y_true, y_pred):
        # Huber损失（对异常值鲁棒）
        huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
        
        # 方向损失：惩罚预测方向错误（仅当batch大小>1时计算）
        batch_size = tf.shape(y_true)[0]
        
        def compute_direction_loss():
            true_diff = y_true[1:] - y_true[:-1]
            pred_diff = y_pred[1:] - y_pred[:-1]
            
            # 使用符号函数并添加epsilon避免除零
            true_direction = tf.sign(true_diff + 1e-10)
            pred_direction = tf.sign(pred_diff + 1e-10)
            
            # 方向匹配度（1表示匹配，0表示不匹配）
            direction_match = tf.cast(tf.equal(true_direction, pred_direction), tf.float32)
            direction_error = 1.0 - tf.reduce_mean(direction_match)
            
            return direction_error
        
        # 只有当batch_size>1时才计算方向损失
        direction_component = tf.cond(
            batch_size > 1,
            compute_direction_loss,
            lambda: 0.0
        )
        
        # 组合损失：80% Huber + 20% 方向（优化版本1最优配置）
        total_loss = 0.80 * huber + 0.20 * direction_component
        
        # 确保输出不为NaN
        return tf.where(tf.math.is_nan(total_loss), huber, total_loss)
    
    # 煤炭价格优化第一轮最优配置:CosineDecay适配learning_rate=0.00015
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=CONFIG['learning_rate'],  # 0.00015
        decay_steps=CONFIG['epochs'] * 50,  # 300*50=15000步
        alpha=0.02  # 最终学习率=0.00015*0.02=3e-06
    )
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule, clipnorm=CONFIG['gradient_clip']),
        loss=directional_loss,
        metrics=['mae', 'mse']
    )
    
    return model

# ============================================================================
# 主系统类
# ============================================================================

class LSTMAttentionCarbonPrediction:
    """LSTM + Attention 碳价格预测系统"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.model = None
        self.history = None
        self.predictions = None
        self.actual_values = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_names = []
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {}
        self.shap_values = None
        self.rf_model = None  # 用于SHAP分析的随机森林模型
        
    def load_data(self, file_path):
        """加载数据"""
        print(f"📊 加载数据文件: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取 Stata 文件
        self.data = pd.read_stata(file_path)
        
        # 转换日期列
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        
        print(f"✅ 数据加载成功")
        print(f"   • 数据形状: {self.data.shape}")
        print(f"   • 时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        print(f"   • 列数: {len(self.data.columns)}")
        
        return self.data
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n🔧 Data Preprocessing...")
        
        df = self.data.copy()
        target = CONFIG['target_column']
        
        # 检查目标列
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")
        
        print(f"   • Original shape: {df.shape}")
        print(f"   • Missing values before cleaning: {df.isnull().sum().sum()}")
        
        # 1. 删除空值比例过高的列（超过80%）
        null_ratio = df.isnull().sum() / len(df)
        cols_to_drop = null_ratio[null_ratio > 0.8].index.tolist()
        if cols_to_drop:
            print(f"   • Dropping columns with >80% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # 2. 删除全为NaN的列
        df = df.dropna(axis=1, how='all')
        
        # 3. 删除空值比例过高的行（超过50%）
        row_null_ratio = df.isnull().sum(axis=1) / len(df.columns)
        rows_to_keep = row_null_ratio <= 0.5
        df = df[rows_to_keep]
        print(f"   • Rows removed: {(~rows_to_keep).sum()}")
        
        # 4. 处理剩余缺失值 - 使用多层插值
        for col in df.columns:
            if df[col].isnull().any():
                # 第一层：线性插值
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                # 第二层：填充剩余缺失值
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
                # 第三层：最后的保险
                if df[col].isnull().any():
                    df[col] = df[col].fillna(0)
        
        print(f"   • Missing values after cleaning: {df.isnull().sum().sum()}")
        
        # 移除无穷大
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        print(f"   • Final shape: {df.shape}")
        
        # 特征工程 - 增强版（使用shift避免数据泄露）
        print("   • Creating lag-shifted technical indicators to prevent data leakage...")
        
        # 第八轮优化:移除shift(1)策略,恢复更多有效样本
        print("   • 第八轮优化:移除shift(1),增加有效样本数...")
        
        # 基础滞后特征 - 减少冗余,仅保留关键lag
        for lag in [1, 3, 7]:
            df[f'price_lag_{lag}'] = df[target].shift(lag)
        
        # 价格变化 - 移除shift(1)
        df['price_return'] = df[target].pct_change()
        
        # 差分特征 - 捕捉趋势变化
        df['price_diff_1'] = df[target].diff(1)
        df['price_diff_7'] = df[target].diff(7)
        df['price_diff_30'] = df[target].diff(30)
        
        # 移动平均 - 移除shift(1)
        for window in [5, 10, 20, 30, 60]:
            df[f'ma_{window}'] = df[target].rolling(window, min_periods=1).mean()
            df[f'ma_{window}_ratio'] = df[target] / (df[f'ma_{window}'] + 1e-10)
        
        # 指数移动平均 - 移除shift(1)
        for span in [12, 26]:
            df[f'ema_{span}'] = df[target].ewm(span=span, adjust=False).mean()
        
        # MACD指标 - 新增
        ema_12 = df[target].ewm(span=12, adjust=False).mean()
        ema_26 = df[target].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 波动率 - 移除shift(1),增加20窗口(用于布林带)
        for window in [7, 14, 20, 30]:  # 增加20
            df[f'volatility_{window}'] = df[target].rolling(window, min_periods=1).std()
            df[f'volatility_{window}_ratio'] = df[f'volatility_{window}'] / (df[target] + 1e-10)
        
        # 布林带 - 新增(现在volatility_20已存在)
        df['bb_upper'] = df['ma_20'] + 2 * df['volatility_20']
        df['bb_lower'] = df['ma_20'] - 2 * df['volatility_20']
        df['bb_position'] = (df[target] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['ma_20'] + 1e-10)
        
        # 价格动量 - 移除shift
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df[target].diff(period)
        
        # 价格变化率 - 移除shift
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df[target].pct_change(period)
        
        # RSI指标 - 移除shift(1)
        for period in [14, 28]:
            delta = df[target].diff()
            gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 第八轮优化:排除冗余特征,保留高质量特征
        leakage_features = [
            'log_carbon_price_hb_ea',  # 目标变量的对数形式
            'log_transactionamount_hb_ea',  # 可能包含未来信息
            'log_transactionamount_hb_ea_sqr',  # 交易量的对数平方
            'ema_12',  # 与MACD重复
            'ema_26',  # 与MACD重复
        ]
        
        feature_cols = [col for col in df.columns 
                       if col != target and col not in leakage_features]
        
        self.feature_names = feature_cols
        
        print(f"   • Excluded leakage features: {[f for f in leakage_features if f in df.columns]}")
        self.processed_data = df
        
        print(f"✅ Data Preprocessing Completed")
        print(f"   • Number of features: {len(self.feature_names)}")
        print(f"   • Data shape: {df.shape}")
        
        return df
    
    def create_sequences(self, data, feature_cols, target_col, seq_length):
        """创建序列数据"""
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            seq_X = data[feature_cols].iloc[i:i+seq_length].values
            seq_y = data[target_col].iloc[i+seq_length]
            
            # 检查是否包含NaN或Inf
            if not (np.isnan(seq_X).any() or np.isinf(seq_X).any() or 
                   np.isnan(seq_y) or np.isinf(seq_y)):
                X.append(seq_X)
                y.append(seq_y)
        
        return np.array(X), np.array(y)
    
    def split_and_scale_data(self):
        """分割并标准化数据"""
        print("\n📊 数据分割和标准化...")
        
        df = self.processed_data
        target = CONFIG['target_column']
        seq_len = CONFIG['sequence_length']
        
        # 创建序列
        X, y = self.create_sequences(df, self.feature_names, target, seq_len)
        
        # 再次检查NaN值
        valid_idx = ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"   • 序列数量: {len(X)}")
        print(f"   • 检查NaN值: {np.isnan(X).any() or np.isnan(y).any()}")
        
        # 时间序列分割
        n = len(X)
        train_size = int(n * (1 - CONFIG['test_size'] - CONFIG['validation_size']))
        val_size = int(n * (1 - CONFIG['test_size']))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]
        
        print(f"   • 训练集: {len(X_train)} 样本")
        print(f"   • 验证集: {len(X_val)} 样本")
        print(f"   • 测试集: {len(X_test)} 样本")
        
        # 标准化特征 - 使用更稳健的方法
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_flat = np.nan_to_num(X_train_flat, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_flat = self.scaler_X.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(X_train.shape)
        
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_flat = np.nan_to_num(X_val_flat, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_flat = self.scaler_X.transform(X_val_flat)
        X_val = X_val_flat.reshape(X_val.shape)
        
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_flat = np.nan_to_num(X_test_flat, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_flat = self.scaler_X.transform(X_test_flat)
        X_test = X_test_flat.reshape(X_test.shape)
        
        # 标准化目标变量
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"✅ 数据分割和标准化完成")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        print("\n🤖 训练 LSTM + Attention 模型...")
        
        n_features = X_train.shape[2]
        
        # 构建模型
        self.model = build_lstm_attention_model(
            sequence_length=CONFIG['sequence_length'],
            n_features=n_features,
            lstm_units=CONFIG['lstm_units'],
            attention_dim=CONFIG['attention_dim']
        )
        
        print("\n模型架构:")
        self.model.summary()
        
        # 第八轮优化:调整回调参数(移除ReduceLROnPlateau避免与CosineDecay冲突)
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=60,  # 增加至60,给更多优化机会
                restore_best_weights=True, 
                verbose=1,
                min_delta=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(OUTPUT_DIR, 'models', 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # 训练
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ 模型训练完成")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """评估模型"""
        print("\n📈 评估模型性能...")
        
        # 预测
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # 反标准化
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 方向准确率
        direction_acc = np.mean(
            np.sign(y_pred[1:] - y_pred[:-1]) == 
            np.sign(y_true[1:] - y_true[:-1])
        ) * 100
        
        self.predictions = y_pred
        self.actual_values = y_true
        
        print(f"   • MSE: {mse:.4f}")
        print(f"   • MAE: {mae:.4f}")
        print(f"   • RMSE: {rmse:.4f}")
        print(f"   • R²: {r2:.4f}")
        print(f"   • MAPE: {mape:.2f}%")
        print(f"   • 方向准确率: {direction_acc:.2f}%")
        
        self.results = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_acc
        }
        
        print("✅ Model Evaluation Completed")
        
        return self.results
    
    def perform_shap_analysis(self, X_train_ml, y_train_ml, X_test_ml):
        """执行SHAP可解释性分析"""
        if not SHAP_AVAILABLE:
            print("\n⚠️ SHAP not installed, skipping interpretability analysis")
            return None
        
        print("\n🔍 Performing SHAP Analysis...")
        
        # 训练随机森林模型用于SHAP分析
        print("   • Training Random Forest for SHAP...")
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_ml, y_train_ml)
        
        # 创建SHAP解释器
        print("   • Creating SHAP explainer...")
        explainer = shap.TreeExplainer(self.rf_model)
        
        # 计算SHAP值（使用测试集样本）
        print("   • Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test_ml[:100])  # 使用前100个样本
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)
        
        print(f"\n   Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"      {row['Feature']:30s}: {row['Importance']:.6f}")
        
        self.shap_values = {
            'values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance,
            'X_test_sample': X_test_ml[:100]
        }
        
        print("\n✅ SHAP Analysis Completed")
        
        return self.shap_values
    
    def create_visualizations(self):
        """生成可视化图表（英文）"""
        print("\n🎨 Generating Visualizations...")
        
        pic_dir = os.path.join(OUTPUT_DIR, 'visualizations')
        
        # 1. Training History
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss Curve', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE Curve', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(pic_dir, f'{self.run_timestamp}_training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Prediction Results
        fig, ax = plt.subplots(figsize=(14, 6))
        
        show_points = min(300, len(self.actual_values))
        
        ax.plot(self.actual_values[-show_points:], label='Actual', linewidth=2)
        ax.plot(self.predictions[-show_points:], label='Predicted', linewidth=2)
        ax.fill_between(range(show_points), 
                        self.actual_values[-show_points:],
                        self.predictions[-show_points:],
                        alpha=0.2)
        ax.set_title(f'Carbon Price Prediction (Last {show_points} Data Points)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Carbon Price')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        textstr = f'R\u00b2 = {self.results["R2"]:.4f}\nRMSE = {self.results["RMSE"]:.4f}\nMAPE = {self.results["MAPE"]:.2f}%'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(pic_dir, f'{self.run_timestamp}_predictions.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Residual Analysis
        residuals = self.actual_values - self.predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(residuals, label='Residuals', alpha=0.7)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].set_title('Prediction Residuals', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Residuals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Residual Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(pic_dir, f'{self.run_timestamp}_residuals.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. SHAP Visualizations
        if self.shap_values is not None and SHAP_AVAILABLE:
            print("   • Creating SHAP visualizations...")
            
            # SHAP Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values['values'], 
                self.shap_values['X_test_sample'],
                feature_names=self.feature_names,
                show=False
            )
            plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(pic_dir, f'{self.run_timestamp}_shap_summary.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # SHAP Bar Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values['values'],
                self.shap_values['X_test_sample'],
                feature_names=self.feature_names,
                plot_type="bar",
                show=False
            )
            plt.title('SHAP Feature Importance (Bar)', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(pic_dir, f'{self.run_timestamp}_shap_bar.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature Importance Comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = self.shap_values['feature_importance'].head(15)
            ax.barh(range(len(top_features)), top_features['Importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Top 15 Important Features (SHAP)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(pic_dir, f'{self.run_timestamp}_feature_importance.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to: {pic_dir}")
    
    def generate_excel_report(self):
        """生成Excel报告"""
        print("\n📊 生成Excel报告...")
        
        excel_path = os.path.join(OUTPUT_DIR, 'reports', 
                                 f'{self.run_timestamp}_carbon_prediction_report.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # 1. 模型性能指标
            metrics_df = pd.DataFrame([
                {'指标': 'MSE', '数值': f"{self.results['MSE']:.6f}"},
                {'指标': 'MAE', '数值': f"{self.results['MAE']:.6f}"},
                {'指标': 'RMSE', '数值': f"{self.results['RMSE']:.6f}"},
                {'指标': 'R²', '数值': f"{self.results['R2']:.6f}"},
                {'指标': 'MAPE(%)', '数值': f"{self.results['MAPE']:.2f}"},
                {'指标': '方向准确率(%)', '数值': f"{self.results['Direction_Accuracy']:.2f}"}
            ])
            metrics_df.to_excel(writer, sheet_name='模型性能指标', index=False)
            
            # 2. 预测结果
            predictions_df = pd.DataFrame({
                '实际值': self.actual_values,
                '预测值': self.predictions,
                '误差': self.actual_values - self.predictions,
                '误差率(%)': (np.abs(self.actual_values - self.predictions) / 
                            self.actual_values) * 100
            })
            predictions_df.to_excel(writer, sheet_name='预测结果', index=False)
            
            # 3. 数据统计
            data_stats = pd.DataFrame([
                {'统计项': '数据总数', '数值': len(self.actual_values)},
                {'统计项': '实际值均值', '数值': f"{self.actual_values.mean():.4f}"},
                {'统计项': '实际值标准差', '数值': f"{self.actual_values.std():.4f}"},
                {'统计项': '实际值最小值', '数值': f"{self.actual_values.min():.4f}"},
                {'统计项': '实际值最大值', '数值': f"{self.actual_values.max():.4f}"},
                {'统计项': '预测值均值', '数值': f"{self.predictions.mean():.4f}"},
                {'统计项': '预测值标准差', '数值': f"{self.predictions.std():.4f}"},
            ])
            data_stats.to_excel(writer, sheet_name='数据统计', index=False)
            
            # 4. 系统配置
            config_df = pd.DataFrame([
                {'配置项': '序列长度', '数值': CONFIG['sequence_length']},
                {'配置项': '训练轮数', '数值': CONFIG['epochs']},
                {'配置项': '批次大小', '数值': CONFIG['batch_size']},
                {'配置项': 'LSTM单元数', '数值': CONFIG['lstm_units']},
                {'配置项': 'Attention维度', '数值': CONFIG['attention_dim']},
                {'配置项': '学习率', '数值': CONFIG['learning_rate']},
                {'配置项': '测试集比例', '数值': CONFIG['test_size']},
            ])
            config_df.to_excel(writer, sheet_name='系统配置', index=False)
        
        print(f"✅ Excel报告已保存到: {excel_path}")
        return excel_path
    
    def generate_text_report(self):
        """生成TXT文本报告"""
        print("\n📄 生成TXT文本报告...")
        
        txt_path = os.path.join(OUTPUT_DIR, 'logs',
                               f'{self.run_timestamp}_analysis_report.txt')
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("碳价格预测系统分析报告 (LSTM + Attention)")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 数据信息
        report_lines.append("📊 数据信息:")
        report_lines.append(f"   • 数据源: data.DTA 文件")
        report_lines.append(f"   • 目标列: {CONFIG['target_column']}")
        report_lines.append(f"   • 数据总数: {len(self.data)}")
        report_lines.append(f"   • 有效序列数: {len(self.actual_values)}")
        report_lines.append(f"   • 时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        report_lines.append("")
        
        # 模型性能
        report_lines.append("🏆 模型性能指标:")
        report_lines.append(f"   • MSE (均方误差): {self.results['MSE']:.6f}")
        report_lines.append(f"   • MAE (平均绝对误差): {self.results['MAE']:.6f}")
        report_lines.append(f"   • RMSE (均方根误差): {self.results['RMSE']:.6f}")
        report_lines.append(f"   • R² (决定系数): {self.results['R2']:.6f}")
        report_lines.append(f"   • MAPE (平均绝对百分比误差): {self.results['MAPE']:.2f}%")
        report_lines.append(f"   • 方向准确率: {self.results['Direction_Accuracy']:.2f}%")
        report_lines.append("")
        
        # 性能评估
        report_lines.append("📈 模型性能评估:")
        if self.results['R2'] > 0.8:
            report_lines.append("   ✅ 模型性能优秀（R² > 0.8），预测精度高")
        elif self.results['R2'] > 0.6:
            report_lines.append("   ✓ 模型性能良好（0.6 < R² ≤ 0.8），预测精度较好")
        elif self.results['R2'] > 0.4:
            report_lines.append("   ⚠ 模型性能一般（0.4 < R² ≤ 0.6），需要优化")
        else:
            report_lines.append("   ❌ 模型性能较差（R² ≤ 0.4），需要重新调整")
        report_lines.append("")
        
        # 数据统计
        report_lines.append("📊 数据统计分析:")
        report_lines.append(f"   实际值:")
        report_lines.append(f"      • 平均值: {self.actual_values.mean():.4f}")
        report_lines.append(f"      • 标准差: {self.actual_values.std():.4f}")
        report_lines.append(f"      • 最小值: {self.actual_values.min():.4f}")
        report_lines.append(f"      • 最大值: {self.actual_values.max():.4f}")
        report_lines.append(f"   预测值:")
        report_lines.append(f"      • 平均值: {self.predictions.mean():.4f}")
        report_lines.append(f"      • 标准差: {self.predictions.std():.4f}")
        report_lines.append(f"      • 最小值: {self.predictions.min():.4f}")
        report_lines.append(f"      • 最大值: {self.predictions.max():.4f}")
        report_lines.append("")
        
        # 模型配置
        report_lines.append("⚙️ 模型配置:")
        report_lines.append(f"   • 序列长度: {CONFIG['sequence_length']}")
        report_lines.append(f"   • LSTM单元数: {CONFIG['lstm_units']}")
        report_lines.append(f"   • Attention维度: {CONFIG['attention_dim']}")
        report_lines.append(f"   • Dropout比率: {CONFIG['dropout_rate']}")
        report_lines.append(f"   • 学习率: {CONFIG['learning_rate']}")
        report_lines.append(f"   • 训练轮数: {CONFIG['epochs']}")
        report_lines.append(f"   • 批次大小: {CONFIG['batch_size']}")
        report_lines.append(f"   • 测试集比例: {CONFIG['test_size']}")
        report_lines.append("")
        
        # 残差分析
        residuals = self.actual_values - self.predictions
        report_lines.append("🔍 残差分析:")
        report_lines.append(f"   • 残差均值: {residuals.mean():.6f}")
        report_lines.append(f"   • 残差标准差: {residuals.std():.6f}")
        report_lines.append(f"   • 残差范围: [{residuals.min():.6f}, {residuals.max():.6f}]")
        report_lines.append("")
        
        # 建议
        report_lines.append("💡 应用建议:")
        report_lines.append("   • 模型使用LSTM + Attention机制，具有较强的时间序列建模能力")
        report_lines.append("   • Attention机制能够自动学习不同时间步的重要程度")
        report_lines.append("   • 建议定期更新模型以适应市场变化")
        report_lines.append("   • 结合市场基本面进行综合判断")
        report_lines.append("   • 监控关键特征的变化趋势")
        report_lines.append("")
        
        # 输出文件信息
        report_lines.append("📁 生成文件清单:")
        report_lines.append(f"   • Excel报告: {os.path.join('outputs/reports', f'{self.run_timestamp}_carbon_prediction_report.xlsx')}")
        report_lines.append(f"   • 训练历史图: {os.path.join('outputs/visualizations', f'{self.run_timestamp}_training_history.png')}")
        report_lines.append(f"   • 预测对比图: {os.path.join('outputs/visualizations', f'{self.run_timestamp}_predictions.png')}")
        report_lines.append(f"   • 残差分析图: {os.path.join('outputs/visualizations', f'{self.run_timestamp}_residuals.png')}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("报告生成完成")
        report_lines.append("=" * 80)
        
        # 保存文件
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ TXT报告已保存到: {txt_path}")
        return txt_path
    
    def run_complete_analysis(self):
        """完整分析流程"""
        print("\n" + "="*80)
        print(" " * 20 + "LSTM + Attention Carbon Price Prediction System")
        print("="*80 + "\n")
        
        try:
            # 1. Load Data
            self.load_data(CONFIG['data_file'])
            
            # 2. Data Preprocessing
            self.preprocess_data()
            
            # 3. Split and Scale Data
            X_train, y_train, X_val, y_val, X_test, y_test = self.split_and_scale_data()
            
            # 4. Train Model
            self.train_model(X_train, y_train, X_val, y_val)
            
            # 5. Evaluate Model
            self.evaluate_model(X_test, y_test)
            
            # 6. SHAP Analysis (使用原始数据训练RF模型)
            if SHAP_AVAILABLE:
                # 准备用于SHAP的数据（不使用序列）
                df = self.processed_data
                target = CONFIG['target_column']
                n = len(df)
                train_size = int(n * (1 - CONFIG['test_size']))
                
                X_train_ml = df[self.feature_names].iloc[:train_size].values
                y_train_ml = df[target].iloc[:train_size].values
                X_test_ml = df[self.feature_names].iloc[train_size:].values
                
                # 处理NaN
                X_train_ml = np.nan_to_num(X_train_ml, nan=0.0)
                y_train_ml = np.nan_to_num(y_train_ml, nan=0.0)
                X_test_ml = np.nan_to_num(X_test_ml, nan=0.0)
                
                self.perform_shap_analysis(X_train_ml, y_train_ml, X_test_ml)
            
            # 7. Create Visualizations
            self.create_visualizations()
            
            # 8. Generate Excel Report
            excel_path = self.generate_excel_report()
            
            # 9. Generate TXT Report
            txt_path = self.generate_text_report()
            
            print("\n" + "="*80)
            print("✅ Complete Analysis Finished Successfully!")
            print("="*80)
            print("\n📁 Generated Files:")
            print(f"   • Excel Report: {excel_path}")
            print(f"   • TXT Report: {txt_path}")
            print(f"   • Visualizations: outputs/visualizations/")
            if SHAP_AVAILABLE and self.shap_values is not None:
                print(f"   • SHAP Analysis: Completed")
            print("\n" + "="*80 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    system = LSTMAttentionCarbonPrediction()
    system.run_complete_analysis()
