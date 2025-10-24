#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
碳价格预测系统 - Baseline五模型对比
包含: RNN, GRU, LSTM, Transformer, AutoFormer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, GRU, SimpleRNN,
    LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, UpSampling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

np.random.seed(42)
tf.random.set_seed(42)

DATA_FILE = '/Users/Jason/Desktop/code/AI/data.dta'
OUTPUT_DIR = '/Users/Jason/Desktop/code/AI/outputs/baseline'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_NAME = f'baseline_{TIMESTAMP}'


def load_and_prepare_data(data_path, target_col='coal_price', seq_len=60):
    """加载和准备数据"""
    print(f"加载数据: {data_path}")
    
    df = pd.read_stata(data_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    print(f"✅ 加载完成: {df.shape}")
    print(f"   列: {df.columns.tolist()}")
    
    # 数据预处理
    print("\n预处理数据...")
    
    # 删除完全为NaN的列
    df = df.dropna(axis=1, how='all')
    
    # 处理剩余的NaN值
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()
    
    # 特征工程（使用min_periods避免NaN）
    for window in [5, 10, 20, 30]:
        df[f'ma_{window}'] = df[target_col].rolling(window, min_periods=1).mean()
    
    # 滞后特征
    for lag in [1, 2, 3, 5, 10]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # 只删除前面lag产生的NaN（最多10行）
    df = df.iloc[10:].reset_index(drop=True)
    
    # 再次处理新产生的NaN
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()
    
    print(f"✅ 预处理完成: {df.shape}")
    
    # 分割数据
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"\n数据分割: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # 准备序列
    feature_cols = [c for c in df.columns if c != target_col]
    
    def make_sequences(data, features, target, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[features].iloc[i-seq_len:i].values)
            y.append(data[target].iloc[i])
        return np.array(X), np.array(y)
    
    X_train, y_train = make_sequences(train_df, feature_cols, target_col, seq_len)
    X_val, y_val = make_sequences(val_df, feature_cols, target_col, seq_len)
    X_test, y_test = make_sequences(test_df, feature_cols, target_col, seq_len)
    
    print(f"\n序列准备: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    
    # 标准化
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    shape_train = X_train.shape
    X_train_scaled = X_scaler.fit_transform(X_train.reshape(-1, shape_train[-1])).reshape(shape_train)
    X_val_scaled = X_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    y_all = np.concatenate([y_train, y_val, y_test])
    y_scaler.fit(y_all.reshape(-1, 1))
    
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test, y_scaler


def build_rnn(n_features, seq_len):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        SimpleRNN(64, return_sequences=True, dropout=0.2),
        SimpleRNN(32, return_sequences=False, dropout=0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_gru(n_features, seq_len):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        GRU(64, return_sequences=True, dropout=0.2),
        GRU(32, return_sequences=False, dropout=0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_lstm(n_features, seq_len):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(64, return_sequences=True, dropout=0.2),
        LSTM(32, return_sequences=False, dropout=0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_transformer(n_features, seq_len):
    inputs = Input(shape=(seq_len, n_features))
    x = Dense(64, activation='relu')(inputs)
    
    attn = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.2)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    
    ffn = Dense(128, activation='relu')(x)
    ffn = Dropout(0.2)(ffn)
    ffn = Dense(64)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization()(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_autoformer(n_features, seq_len):
    inputs = Input(shape=(seq_len, n_features))
    
    x = Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(2, padding='same')(x)
    x = UpSampling1D(2)(x)
    
    attn = MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.2)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


def train_model(model, name, X_train, X_val, y_train, y_val, X_test, y_test, y_scaler, epochs=150):
    """训练单个模型"""
    print(f"\n{'='*70}")
    print(f"训练 {name}...")
    print(f"{'='*70}")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, f'{RUN_NAME}_{name.lower()}_best.h5'),
                       monitor='val_loss', save_best_only=True, verbose=0)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 预测和评估
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_scaled_clipped = np.clip(y_pred_scaled, 0, 1)
    y_pred = y_scaler.inverse_transform(y_pred_scaled_clipped).flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    
    direction_acc = np.mean(
        np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_test[1:] - y_test[:-1])
    ) * 100
    
    print(f"\n{name}结果:")
    print(f"  R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, Direction={direction_acc:.2f}%")
    
    return {
        'model': model,
        'history': history,
        'y_pred': y_pred,
        'y_true': y_test,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Direction_Accuracy': direction_acc
    }


def main():
    print("\n🌍 " + "="*60)
    print("  碳价格预测 - Baseline五模型对比")
    print("  (RNN, GRU, LSTM, Transformer, AutoFormer)")
    print("="*60 + " 🌍\n")
    
    # 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = load_and_prepare_data(DATA_FILE)
    
    n_features = X_train.shape[2]
    seq_len = X_train.shape[1]
    
    # 构建和训练模型
    models_config = {
        'RNN': build_rnn,
        'GRU': build_gru,
        'LSTM': build_lstm,
        'Transformer': build_transformer,
        'AutoFormer': build_autoformer
    }
    
    results = {}
    for model_name, builder in models_config.items():
        model = builder(n_features, seq_len)
        result = train_model(model, model_name, X_train, X_val, y_train, y_val, X_test, y_test, y_scaler)
        results[model_name] = result
    
    # 生成对比表
    print("\n" + "="*70)
    print("模型性能对比")
    print("="*70)
    
    results_list = []
    for name, metrics in results.items():
        results_list.append({
            '模型': name,
            'R²': metrics['R²'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE(%)': metrics['MAPE'],
            '方向准确率(%)': metrics['Direction_Accuracy']
        })
    
    results_df = pd.DataFrame(results_list).sort_values('R²', ascending=False)
    
    print("\n排序结果（按R²排序）:")
    print(results_df.to_string(index=False))
    
    # 保存结果
    csv_path = os.path.join(OUTPUT_DIR, f'{RUN_NAME}_baseline_results.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存到: {csv_path}")
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('五大Baseline模型性能对比', fontsize=16, fontweight='bold')
    
    metrics = ['R²', 'RMSE', 'MAE', 'MAPE(%)', '方向准确率(%)']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(results_df['模型'], results_df[metric], alpha=0.7, color='steelblue')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, results_df[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.delaxes(axes[1, 2])
    plt.tight_layout()
    
    pic_path = os.path.join(OUTPUT_DIR, f'{RUN_NAME}_baseline_comparison.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存到: {pic_path}")
    plt.show()
    
    # 绘制预测结果
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.ravel()
    
    for idx, (name, metrics) in enumerate(results.items()):
        ax = axes[idx]
        
        y_true = metrics['y_true'][-200:]
        y_pred = metrics['y_pred'][-200:]
        
        ax.plot(y_true, label='实际值', linewidth=2, alpha=0.8, color='blue')
        ax.plot(y_pred, label='预测值', linewidth=2, alpha=0.8, color='red')
        
        ax.set_title(f'{name} (R²={metrics["R²"]:.4f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('时间步')
        ax.set_ylabel('价格')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.delaxes(axes[5])
    plt.tight_layout()
    
    pic_path = os.path.join(OUTPUT_DIR, f'{RUN_NAME}_baseline_predictions.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"✅ 预测结果图已保存到: {pic_path}")
    plt.show()
    
    print("\n🎉 程序执行完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
