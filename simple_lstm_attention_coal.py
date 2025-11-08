#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)

CONFIG = {
    'data_file': 'data.dta',
    'target_column': 'coal_price',
    'sequence_length': 60,  # 🔥 增加到60以捕捉更长依赖
    'test_size': 0.2,
    'validation_size': 0.1,
    'epochs': 300,  # 🔥 增加epoch防止早停过早
    'batch_size': 16,  # 🔥 减小batch size增加梯度更新频率
    'learning_rate': 0.0005,  # 🔥 降低学习率提升稳定性
    'lstm_units': [128, 64],  # 🔥 改为双层LSTM
    'attention_dim': 64,  # 🔥 增加attention容量
    'lstm_dropout': 0.3,
    'lstm_recurrent_dropout': 0.2,
    'dropout_rate': 0.5,  # 🔥 增强正则化
    'dense_units_1': 64,  # 🔥 增加Dense层容量
    'dense_units_2': 32,
    'scaler_type': 'minmax',
    'use_residual': True,  # 🔥 新增残差连接选项
}

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_simple_attention(input_tensor, attention_dim):
    query = layers.Dense(attention_dim, name='attention_query')(input_tensor)
    key = layers.Dense(attention_dim, name='attention_key')(input_tensor)
    value = layers.Dense(attention_dim, name='attention_value')(input_tensor)
    scores = layers.Dot(axes=[2, 2])([query, key])
    scores = layers.Lambda(lambda x: x / tf.math.sqrt(tf.cast(attention_dim, tf.float32)))(scores)
    attention_weights = layers.Softmax(axis=-1, name='attention_weights')(scores)
    context = layers.Dot(axes=[2, 1])([attention_weights, value])
    context_vector = layers.GlobalAveragePooling1D(name='attention_pooling')(context)
    return context_vector


def build_simple_lstm_attention(sequence_length, n_features):
    inputs = layers.Input(shape=(sequence_length, n_features))
    
    # 🔥 多层LSTM架构
    lstm_units_list = CONFIG['lstm_units'] if isinstance(CONFIG['lstm_units'], list) else [CONFIG['lstm_units']]
    x = inputs
    
    for i, units in enumerate(lstm_units_list):
        return_seq = (i < len(lstm_units_list) - 1)  # 最后一层不返回序列
        x = layers.LSTM(
            units,
            return_sequences=True,  # 🔥 所有层都返回序列用于Attention
            dropout=CONFIG['lstm_dropout'],
            recurrent_dropout=CONFIG['lstm_recurrent_dropout'],
            name=f'lstm_layer_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
    
    lstm_out = x  # 保存最后的LSTM输出
    
    # Attention机制
    attention_out = create_simple_attention(lstm_out, CONFIG['attention_dim'])
    
    # 🔥 残差连接优化
    lstm_pooled = layers.GlobalAveragePooling1D(name='lstm_pooling')(lstm_out)
    if lstm_pooled.shape[-1] != CONFIG['attention_dim']:
        lstm_pooled = layers.Dense(CONFIG['attention_dim'], name='residual_projection')(lstm_pooled)
    
    combined = layers.Add(name='residual_connection')([lstm_pooled, attention_out])
    combined = layers.LayerNormalization(epsilon=1e-6, name='layer_norm')(combined)
    
    # 🔥 增强的Dense层
    dense = layers.Dense(CONFIG['dense_units_1'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(CONFIG['dropout_rate'])(dense)
    
    dense = layers.Dense(CONFIG['dense_units_2'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense)
    dense = layers.Dropout(CONFIG['dropout_rate'] * 0.5)(dense)
    
    outputs = layers.Dense(1)(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # 🔥 使用固定学习率，配合ReduceLROnPlateau动态调整
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate'], clipnorm=1.0),  # 🔥 梯度裁剪
        loss='mse',
        metrics=['mae']
    )
    return model

class SimpleCoalPricePrediction:
    
    def __init__(self):
        self.data = None
        self.model = None
        self.history = None
        self.scaler_X = None  # 将在prepare_data中初始化
        self.scaler_y = None  # 将在prepare_data中初始化
        self.feature_names = []
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.shap_values = None
        self.rf_model = None
        
    def load_data(self, file_path):
        """加载数据"""
        print(f"📊 加载数据文件: {file_path}")
        
        self.data = pd.read_stata(file_path)
        
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        original_shape = self.data.shape
        
        # 🔥 使用2017-2024年数据，排除特定异常时段
        self.data = self.data[(self.data.index.year >= 2017) & (self.data.index.year <= 2024)]
        
        # 排除异常时段1: 2021.6 到 2022.1
        exclude_1_start = pd.Timestamp('2021-06-01')
        exclude_1_end = pd.Timestamp('2022-01-31')
        exclude_condition_1 = (self.data.index >= exclude_1_start) & (self.data.index <= exclude_1_end)
        
        # 排除异常时段2: 2022.3 到 2022.10
        exclude_2_start = pd.Timestamp('2022-03-01')
        exclude_2_end = pd.Timestamp('2022-10-31')
        exclude_condition_2 = (self.data.index >= exclude_2_start) & (self.data.index <= exclude_2_end)
        
        # 应用排除条件（两个时段的并集）
        self.data = self.data[~(exclude_condition_1 | exclude_condition_2)]
        print(f"✅ 数据加载成功")
        print(f"   • 原始: {original_shape}, 筛选后: {self.data.shape}")
        print(f"   • 时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        return self.data
    
    def create_enhanced_features(self, df, target):
        print("   • 创建特征...")
        # 🔥 保留 oil_price 和 log_carbon_price 相关特征
        original_features = [
            'oil_price',
            'log_oil_price',
            'log_oil_price_sqr',
            'log_carbon_price',  
        ]
        available_features = [f for f in original_features if f in df.columns]
        print(f"      ✅ 找到 {len(available_features)} 个原始特征")
        df = df[[target] + available_features]
        
        # 🔥 为 coal_price 添加技术指标
        print("      • 添加coal_price技术指标...")
        
        # MA - 移动平均 (2, 5, 10日)
        for window in [2, 5, 10]:
            df[f'{target}_ma{window}'] = df[target].rolling(window=window, min_periods=1).mean()
        
        # MACD - 移动平均收敛发散指标
        exp12 = df[target].ewm(span=12, adjust=False).mean()
        exp26 = df[target].ewm(span=26, adjust=False).mean()
        df[f'{target}_macd'] = exp12 - exp26
        df[f'{target}_macd_signal'] = df[f'{target}_macd'].ewm(span=9, adjust=False).mean()
        df[f'{target}_macd_hist'] = df[f'{target}_macd'] - df[f'{target}_macd_signal']
        
        # 动量指标 - Momentum (10日)
        df[f'{target}_momentum'] = df[target].diff(10)
        
        # EMA - 指数移动平均 (12, 26日)
        df[f'{target}_ema12'] = df[target].ewm(span=12, adjust=False).mean()
        df[f'{target}_ema26'] = df[target].ewm(span=26, adjust=False).mean()
        
        print(f"      ✅ 已添加 coal_price 技术指标: MA(2,5,10) + MACD(3个) + Momentum + EMA(2个)")
        
        # 为其他特征添加MA5
        print("      • 添加其他特征MA5...")
        ma_count = 0
        for feature in available_features:
            ma_col_name = f'{feature}_ma5'
            df[ma_col_name] = df[feature].rolling(window=5, min_periods=1).mean()
            ma_count += 1
        
        print(f"      ✅ 已添加 {ma_count} 个MA5移动平均特征")
        
        return df
    
    def remove_outliers(self, df, target):
        """使用MAD方法移除异常值"""
        if not CONFIG['remove_outliers']:
            return df
        
        print("   • 移除异常值...")
        original_len = len(df)
        
        # 对目标变量使用MAD方法
        median = df[target].median()
        mad = np.median(np.abs(df[target] - median))
        threshold = CONFIG['outlier_threshold']
        
        # 计算修正的z-score
        modified_z_scores = 0.6745 * (df[target] - median) / (mad + 1e-10)
        
        # 移除异常值
        df = df[np.abs(modified_z_scores) < threshold]
        
        removed = original_len - len(df)
        print(f"      移除了 {removed} 个异常值 ({removed/original_len*100:.2f}%)")
        
        return df
    
    def select_features(self, df, target):
        """基于互信息的特征选择"""
        if not CONFIG['feature_selection']:
            return df
        
        print("   • 执行特征选择...")
        
        feature_cols = [col for col in df.columns if col != target]
        
        if len(feature_cols) <= CONFIG['top_features']:
            print(f"      特征数量 ({len(feature_cols)}) <= 阈值 ({CONFIG['top_features']})，保留所有特征")
            return df
        
        # 准备数据
        X = df[feature_cols].fillna(0)
        y = df[target]
        
        # 计算互信息
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # 选择top特征
        top_features = feature_importance.head(CONFIG['top_features'])['feature'].tolist()
        
        print(f"      从 {len(feature_cols)} 个特征中选择了 {len(top_features)} 个最重要特征")
        print(f"      Top 5 特征: {top_features[:5]}")
        
        # 保留目标列和选中的特征
        selected_cols = [target] + top_features
        
        return df[selected_cols]
    
    def preprocess_data(self):
        """增强的数据预处理流程"""
        print("\n🔧 数据预处理...")
        
        df = self.data.copy()
        target = CONFIG['target_column']
        
        # 检查目标列
        if target not in df.columns:
            raise ValueError(f"目标列 '{target}' 不存在")
        
        print(f"   • 原始形状: {df.shape}")
        
        df = df.dropna(axis=1, how='all')
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        df = self.create_enhanced_features(df, target)
        df = df.dropna()
        self.feature_names = [col for col in df.columns if col != target]
        print(f"✅ 数据预处理完成: {len(self.feature_names)} 特征, {df.shape}")
        return df
    
    def create_sequences(self, data, feature_cols, target_col, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            seq_X = data[feature_cols].iloc[i:i+seq_length].values
            seq_y = data[target_col].iloc[i+seq_length]
            if not (np.isnan(seq_X).any() or np.isnan(seq_y)):
                X.append(seq_X)
                y.append(seq_y)
        return np.array(X), np.array(y)
    

    def prepare_data(self, df):
        print("\n📊 准备训练数据...")
        target = CONFIG['target_column']
        seq_len = CONFIG['sequence_length']
        X, y = self.create_sequences(df, self.feature_names, target, seq_len)
        print(f"   • 序列数量: {len(X)}")
        n = len(X)
        train_size = int(n * (1 - CONFIG['test_size'] - CONFIG['validation_size']))
        val_size = int(n * (1 - CONFIG['test_size']))
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]
        print(f"   • 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
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
        print("\n🤖 训练单层 LSTM + Attention 模型...")
        self.model = build_simple_lstm_attention(
            sequence_length=CONFIG['sequence_length'],
            n_features=X_train.shape[2]
        )
        lstm_config = 'x'.join(map(str, CONFIG['lstm_units'])) if isinstance(CONFIG['lstm_units'], list) else str(CONFIG['lstm_units'])
        print(f"\n模型架构: LSTM=[{lstm_config}], Attention={CONFIG['attention_dim']}, Dense=[{CONFIG['dense_units_1']},{CONFIG['dense_units_2']}]")
        self.model.summary()
        # 🔥 增强的回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1, min_delta=1e-4),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1)
        ]
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
        print("\n📈 评估模型性能...")
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        direction_acc = np.mean(np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])) * 100
        print(f"\n结果: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, 方向准确率={direction_acc:.2f}%")
        return {
            'y_true': y_true, 'y_pred': y_pred, 'R2': r2, 'RMSE': rmse, 
            'MAE': mae, 'MAPE': mape, 'Direction_Accuracy': direction_acc
        }
    
    def perform_shap_analysis(self, X_train_ml, y_train_ml, X_test_ml):
        if not SHAP_AVAILABLE:
            return None
        print("\n🔍 执行SHAP分析...")
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_train_ml, y_train_ml)
        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer.shap_values(X_test_ml[:100])
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)
        print(f"\n   Top 5: {feature_importance.head(5)['Feature'].tolist()}")
        self.shap_values = {
            'values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance,
            'X_test_sample': X_test_ml[:100]
        }
        print("✅ SHAP分析完成")
        return self.shap_values
    
    def visualize(self, results):
        print("\n🎨 生成可视化图表...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_training.png'), dpi=300)
        plt.close()
        fig, ax = plt.subplots(figsize=(14, 6))
        y_true = results['y_true']
        y_pred = results['y_pred']
        show_points = min(300, len(y_true))
        ax.plot(y_true[-show_points:], label='Actual', linewidth=2)
        ax.plot(y_pred[-show_points:], label='Predicted', linewidth=2)
        ax.set_title(f'Coal Price Prediction (Last {show_points} Points)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_predictions.png'), dpi=300)
        plt.close()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_title('Prediction Scatter Plot', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_scatter.png'), dpi=300)
        plt.close()
        if self.shap_values is not None and SHAP_AVAILABLE:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values['values'], self.shap_values['X_test_sample'],
                            feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_shap_summary.png'), dpi=300)
            plt.close()
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values['values'], self.shap_values['X_test_sample'],
                            feature_names=self.feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_shap_bar.png'), dpi=300)
            plt.close()
        print(f"✅ 可视化图表已保存")
    
    def save_report(self, results):
        print("\n📊 保存分析报告...")
        report_path = os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_coal_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("简单单层 LSTM + Attention 煤炭价格预测报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            lstm_config = 'x'.join(map(str, CONFIG['lstm_units'])) if isinstance(CONFIG['lstm_units'], list) else str(CONFIG['lstm_units'])
            f.write(f"📊 模型配置: LSTM=[{lstm_config}], Attention={CONFIG['attention_dim']}, Seq={CONFIG['sequence_length']}\n")
            f.write(f"📈 模型性能: R²={results['R2']:.4f}, RMSE={results['RMSE']:.4f}, MAE={results['MAE']:.4f}, MAPE={results['MAPE']:.2f}%\n")
            f.write("=" * 80 + "\n")
        print(f"✅ 报告已保存")
    
    def run(self):
        print("\n" + "="*80)
        print(" " * 15 + "简单单层 LSTM + Attention 煤炭价格预测系统")
        print("="*80 + "\n")
        self.load_data(CONFIG['data_file'])
        df = self.preprocess_data()
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df)
        self.train(X_train, y_train, X_val, y_val)
        results = self.evaluate(X_test, y_test)
        if SHAP_AVAILABLE:
            target = CONFIG['target_column']
            n = len(df)
            train_size = int(n * (1 - CONFIG['test_size']))
            X_train_ml = np.nan_to_num(df[self.feature_names].iloc[:train_size].values)
            y_train_ml = np.nan_to_num(df[target].iloc[:train_size].values)
            X_test_ml = np.nan_to_num(df[self.feature_names].iloc[train_size:].values)
            self.perform_shap_analysis(X_train_ml, y_train_ml, X_test_ml)
        self.visualize(results)
        self.save_report(results)
        print("\n" + "="*80)
        print("✅ 分析完成!")
        print("="*80 + "\n")

if __name__ == '__main__':
    predictor = SimpleCoalPricePrediction()
    predictor.run()
