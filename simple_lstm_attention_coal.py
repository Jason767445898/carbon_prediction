#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的单层 LSTM + Attention 煤炭价格预测模型
使用 data.dta 数据文件预测 coal_price
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
# 配置参数 - 简化配置
# ============================================================================

CONFIG = {
    'data_file': 'data.dta',
    'target_column': 'coal_price',
    'sequence_length': 60,
    'test_size': 0.2,
    'validation_size': 0.1,
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 0.001,
    'lstm_units': 128,           # 单层LSTM
    'attention_dim': 128,
    'dropout_rate': 0.3,
}

# 输出目录
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 模型构建
# ============================================================================

def create_simple_attention(input_tensor, attention_dim):
    """
    创建简单的 Attention 层
    
    参数:
        input_tensor: shape=(batch_size, sequence_length, features)
        attention_dim: attention维度
    
    返回:
        context_vector: shape=(batch_size, attention_dim)
    """
    # Query, Key, Value 转换
    query = layers.Dense(attention_dim, name='attention_query')(input_tensor)
    key = layers.Dense(attention_dim, name='attention_key')(input_tensor)
    value = layers.Dense(attention_dim, name='attention_value')(input_tensor)
    
    # 计算 attention scores
    scores = layers.Dot(axes=[2, 2])([query, key])
    scores = layers.Lambda(lambda x: x / tf.math.sqrt(tf.cast(attention_dim, tf.float32)))(scores)
    
    # Softmax 获得 attention weights
    attention_weights = layers.Softmax(axis=-1, name='attention_weights')(scores)
    
    # 应用 attention weights
    context = layers.Dot(axes=[2, 1])([attention_weights, value])
    
    # 全局平均池化
    context_vector = layers.GlobalAveragePooling1D(name='attention_pooling')(context)
    
    return context_vector


def build_simple_lstm_attention(sequence_length, n_features):
    """
    构建简单的单层 LSTM + Attention 模型（带残差连接）
    """
    inputs = layers.Input(shape=(sequence_length, n_features))
    
    # 单层 LSTM
    lstm_out = layers.LSTM(
        CONFIG['lstm_units'], 
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.1
    )(inputs)
    lstm_out = layers.BatchNormalization()(lstm_out)
    
    # Attention 层
    attention_out = create_simple_attention(lstm_out, CONFIG['attention_dim'])
    
    # 残差连接：将LSTM聚合表示与attention输出相加
    lstm_pooled = layers.GlobalAveragePooling1D(name='lstm_pooling')(lstm_out)
    
    # 维度匹配：如果attention_dim与LSTM单元数不同，需要投影
    if CONFIG['attention_dim'] != CONFIG['lstm_units']:
        lstm_pooled = layers.Dense(CONFIG['attention_dim'], name='residual_projection')(lstm_pooled)
    
    # 残差连接：Add层
    combined = layers.Add(name='residual_connection')([lstm_pooled, attention_out])
    
    # Layer Normalization
    combined = layers.LayerNormalization(epsilon=1e-6, name='layer_norm')(combined)
    
    # 全连接层
    dense = layers.Dense(64, activation='relu')(combined)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(CONFIG['dropout_rate'])(dense)
    
    dense = layers.Dense(32, activation='relu')(dense)
    dense = layers.Dropout(CONFIG['dropout_rate'] / 2)(dense)
    
    # 输出层
    outputs = layers.Dense(1)(dense)
    
    # 编译模型
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ============================================================================
# 数据处理
# ============================================================================

class SimpleCoalPricePrediction:
    """简单的煤炭价格预测系统"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.history = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_names = []
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.shap_values = None
        self.rf_model = None
        
    def load_data(self, file_path):
        """加载数据"""
        print(f"📊 加载数据文件: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取 Stata 文件
        self.data = pd.read_stata(file_path)
        
        # 转换日期列
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        
        print(f"✅ 数据加载成功")
        print(f"   • 数据形状: {self.data.shape}")
        print(f"   • 时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        
        return self.data
    
    def preprocess_data(self):
        """数据预处理 - 简化版本"""
        print("\n🔧 数据预处理...")
        
        df = self.data.copy()
        target = CONFIG['target_column']
        
        # 检查目标列
        if target not in df.columns:
            raise ValueError(f"目标列 '{target}' 不存在")
        
        print(f"   • 原始形状: {df.shape}")
        
        # 处理缺失值
        df = df.dropna(axis=1, how='all')
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 移除无穷大
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())
        
        # 简单的特征工程
        print("   • 创建技术指标...")
        
        # 滞后特征
        for lag in [1, 3, 7]:
            df[f'price_lag_{lag}'] = df[target].shift(lag)
        
        # 移动平均
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df[target].rolling(window, min_periods=1).mean()
        
        # 波动率
        for window in [7, 14]:
            df[f'volatility_{window}'] = df[target].rolling(window, min_periods=1).std()
        
        # 价格变化率
        df['price_return'] = df[target].pct_change()
        
        # 删除包含 NaN 的行
        df = df.dropna()
        
        # 选择特征列
        self.feature_names = [col for col in df.columns if col != target]
        
        print(f"✅ 数据预处理完成")
        print(f"   • 特征数量: {len(self.feature_names)}")
        print(f"   • 数据形状: {df.shape}")
        
        return df
    
    def create_sequences(self, data, feature_cols, target_col, seq_length):
        """创建序列数据"""
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            seq_X = data[feature_cols].iloc[i:i+seq_length].values
            seq_y = data[target_col].iloc[i+seq_length]
            
            if not (np.isnan(seq_X).any() or np.isnan(seq_y)):
                X.append(seq_X)
                y.append(seq_y)
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df):
        """准备训练数据"""
        print("\n📊 准备训练数据...")
        
        target = CONFIG['target_column']
        seq_len = CONFIG['sequence_length']
        
        # 创建序列
        X, y = self.create_sequences(df, self.feature_names, target, seq_len)
        
        print(f"   • 序列数量: {len(X)}")
        
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
        
        # 标准化
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_flat = self.scaler_X.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(X_train.shape)
        
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_flat = self.scaler_X.transform(X_val_flat)
        X_val = X_val_flat.reshape(X_val.shape)
        
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_flat = self.scaler_X.transform(X_test_flat)
        X_test = X_test_flat.reshape(X_test.shape)
        
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"✅ 数据准备完成")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        print("\n🤖 训练单层 LSTM + Attention 模型...")
        
        n_features = X_train.shape[2]
        
        # 构建模型
        self.model = build_simple_lstm_attention(
            sequence_length=CONFIG['sequence_length'],
            n_features=n_features
        )
        
        print(f"\n模型架构:")
        print(f"   • 单层 LSTM: {CONFIG['lstm_units']} units")
        print(f"   • Attention维度: {CONFIG['attention_dim']}")
        print(f"   • 残差连接: 已启用")
        self.model.summary()
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=30,
                restore_best_weights=True, 
                verbose=1
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
    
    def evaluate(self, X_test, y_test):
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
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # 方向准确率
        direction_acc = np.mean(
            np.sign(y_pred[1:] - y_pred[:-1]) == 
            np.sign(y_true[1:] - y_true[:-1])
        ) * 100
        
        print(f"\n模型评估结果:")
        print(f"   • R² (决定系数): {r2:.4f}")
        print(f"   • RMSE (均方根误差): {rmse:.4f}")
        print(f"   • MAE (平均绝对误差): {mae:.4f}")
        print(f"   • MAPE: {mape:.2f}%")
        print(f"   • 方向准确率: {direction_acc:.2f}%")
        
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Direction_Accuracy': direction_acc
        }
        
        return results
    
    def perform_shap_analysis(self, X_train_ml, y_train_ml, X_test_ml):
        """执行SHAP可解释性分析"""
        if not SHAP_AVAILABLE:
            print("\n⚠️ SHAP未安装，跳过可解释性分析")
            return None
        
        print("\n🔍 执行SHAP分析...")
        
        # 训练随机森林模型用于SHAP分析
        print("   • 训练随机森林模型...")
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_ml, y_train_ml)
        
        # 创建SHAP解释器
        print("   • 创建SHAP解释器...")
        explainer = shap.TreeExplainer(self.rf_model)
        
        # 计算SHAP值（使用测试集样本）
        print("   • 计算SHAP值...")
        shap_values = explainer.shap_values(X_test_ml[:100])  # 使用前100个样本
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)
        
        print(f"\n   Top 10 重要特征:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"      {row['Feature']:30s}: {row['Importance']:.6f}")
        
        self.shap_values = {
            'values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance,
            'X_test_sample': X_test_ml[:100]
        }
        
        print("\n✅ SHAP分析完成")
        
        return self.shap_values
    
    def visualize(self, results):
        """生成可视化"""
        print("\n🎨 生成可视化图表...")
        
        # 1. 训练历史
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_training.png'), dpi=300)
        plt.close()
        
        # 2. 预测结果
        fig, ax = plt.subplots(figsize=(14, 6))
        
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        show_points = min(300, len(y_true))
        
        ax.plot(y_true[-show_points:], label='Actual', linewidth=2, marker='o', markersize=3)
        ax.plot(y_pred[-show_points:], label='Predicted', linewidth=2, marker='s', markersize=3)
        ax.fill_between(range(show_points), 
                        y_true[-show_points:],
                        y_pred[-show_points:],
                        alpha=0.2, color='gray')
        ax.set_title(f'Coal Price Prediction (Last {show_points} Points)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Coal Price')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        textstr = f'R² = {results["R2"]:.4f}\nRMSE = {results["RMSE"]:.4f}\nMAPE = {results["MAPE"]:.2f}%'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_predictions.png'), dpi=300)
        plt.close()
        
        # 3. 散点图
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # 理想预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Coal Price', fontsize=12)
        ax.set_ylabel('Predicted Coal Price', fontsize=12)
        ax.set_title('Prediction Scatter Plot', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_scatter.png'), dpi=300)
        plt.close()
        
        # 4. SHAP可视化
        if self.shap_values is not None and SHAP_AVAILABLE:
            print("   • 创建SHAP可视化...")
            
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
            plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_shap_summary.png'),
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
            plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_shap_bar.png'),
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
            plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_feature_importance.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ 可视化图表已保存到: {OUTPUT_DIR}")
    
    def save_report(self, results):
        """保存报告"""
        print("\n📊 保存分析报告...")
        
        report_path = os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_coal_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("简单单层 LSTM + Attention 煤炭价格预测报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 模型配置:\n")
            f.write(f"   • 目标变量: {CONFIG['target_column']}\n")
            f.write(f"   • 序列长度: {CONFIG['sequence_length']}\n")
            f.write(f"   • LSTM单元数: {CONFIG['lstm_units']}\n")
            f.write(f"   • Attention维度: {CONFIG['attention_dim']}\n")
            f.write(f"   • 学习率: {CONFIG['learning_rate']}\n")
            f.write(f"   • 批次大小: {CONFIG['batch_size']}\n\n")
            
            f.write("📈 模型性能:\n")
            f.write(f"   • R² (决定系数): {results['R2']:.4f}\n")
            f.write(f"   • RMSE (均方根误差): {results['RMSE']:.4f}\n")
            f.write(f"   • MAE (平均绝对误差): {results['MAE']:.4f}\n")
            f.write(f"   • MAPE: {results['MAPE']:.2f}%\n")
            f.write(f"   • 方向准确率: {results['Direction_Accuracy']:.2f}%\n\n")
            
            f.write("📁 生成文件:\n")
            f.write(f"   • 训练历史图: {self.run_timestamp}_training.png\n")
            f.write(f"   • 预测对比图: {self.run_timestamp}_predictions.png\n")
            f.write(f"   • 散点分布图: {self.run_timestamp}_scatter.png\n")
            if self.shap_values is not None:
                f.write(f"   • SHAP摘要图: {self.run_timestamp}_shap_summary.png\n")
                f.write(f"   • SHAP条形图: {self.run_timestamp}_shap_bar.png\n")
                f.write(f"   • 特征重要性图: {self.run_timestamp}_feature_importance.png\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"✅ 报告已保存到: {report_path}")
    
    def run(self):
        """完整运行流程"""
        print("\n" + "="*80)
        print(" " * 15 + "简单单层 LSTM + Attention 煤炭价格预测系统")
        print("="*80 + "\n")
        
        # 1. 加载数据
        self.load_data(CONFIG['data_file'])
        
        # 2. 预处理
        df = self.preprocess_data()
        
        # 3. 准备数据
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df)
        
        # 4. 训练模型
        self.train(X_train, y_train, X_val, y_val)
        
        # 5. 评估模型
        results = self.evaluate(X_test, y_test)
        
        # 6. SHAP分析
        if SHAP_AVAILABLE:
            # 准备用于SHAP的数据（不使用序列）
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
        
        # 7. 可视化
        self.visualize(results)
        
        # 8. 保存报告
        self.save_report(results)
        
        print("\n" + "="*80)
        print("✅ 分析完成!")
        print("="*80 + "\n")


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == '__main__':
    predictor = SimpleCoalPricePrediction()
    predictor.run()
