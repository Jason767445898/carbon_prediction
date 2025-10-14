#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数调优脚本 v2.0 - 使用改进的数据预处理
用于调整LSTM和Transformer模型的超参数
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/Users/Jason/Desktop/code/AI')

from carbon_price_prediction import CarbonPricePredictionSystem

def create_tuning_log():
    """创建参数调优日志文件"""
    log_content = [
        "=" * 70,
        "碳价格预测模型参数调优记录 v3.0（第三轮优化）",
        "=" * 70,
        f"调优开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "📊 第二轮最佳结果:",
        "- LSTM: R²=0.8768, RMSE=36.37, batch_size=8",
        "- Transformer: 仍然严重过拟合(R²=-0.9251)",
        "",
        "🎯 第三轮优化目标:",
        "- LSTM: 在R²=0.8768基础上突破至R²>0.90，接近RandomForest(0.943)",
        "- Transformer: 采用激进简化策略，目标达到R²>0.3",
        "- 探索更小batch_size和网络结构优化",
        "",
        "🔧 优化策略:",
        "- LSTM: 微调dropout、epochs、网络宽度",
        "- Transformer: 极简网络(d_model=16-32, 1层), 极高dropout(0.5-0.6)",
        "",
        "=" * 70,
        ""
    ]
    
    log_file = 'parameter_tuning_v3.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_content))
    
    print(f"已创建参数调优日志文件: {log_file}")
    return log_file

def log_tuning_result(log_file, config, results, notes=""):
    """记录调优结果"""
    log_entry = [
        f"调优时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "配置参数:",
        f"  LSTM配置: {config.get('lstm_config', 'N/A')}",
        f"  Transformer配置: {config.get('transformer_config', 'N/A')}",
        "",
        "模型性能:",
    ]
    
    for model_name, metrics in results.items():
        log_entry.append(f"  {model_name}:")
        for metric, value in metrics.items():
            if metric not in ['predictions', 'actual']:
                log_entry.append(f"    {metric}: {value:.4f}")
    
    if notes:
        log_entry.append(f"\n备注: {notes}")
    
    log_entry.append("-" * 70)
    log_entry.append("")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_entry))

def tune_lstm_parameters(log_file):
    """调优LSTM模型参数"""
    print("\n" + "=" * 70)
    print("🔍 开始LSTM模型参数调优")
    print("=" * 70)
    
    # 基础配置
    base_config = {
        'target_column': 'coal_price',
        'sequence_length': 60,
        'test_size': 0.2,
        'validation_size': 0.1,
        'transformer_config': {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 256,
            'dropout': 0.3,
            'epochs': 50
        }
    }
    
    # LSTM参数组合 - 基于batch_size=8的最佳结果(R²=0.8768)进一步优化
    lstm_configs = [
        # 1. 最佳基线配置
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 8,
            'name': '最佳基线(batch=8,R²=0.8768)'
        },
        # 2. 降低dropout提高拟合能力
        {
            'units': [64, 32],
            'dropout': 0.15,
            'epochs': 100,
            'batch_size': 8,
            'name': 'batch=8,dropout=0.15'
        },
        # 3. 进一步降低dropout
        {
            'units': [64, 32],
            'dropout': 0.1,
            'epochs': 100,
            'batch_size': 8,
            'name': 'batch=8,dropout=0.1'
        },
        # 4. 增加训练轮数
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 150,
            'batch_size': 8,
            'name': 'batch=8,epochs=150'
        },
        # 5. 增加网络宽度
        {
            'units': [96, 48],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 8,
            'name': 'batch=8,units=[96,48]'
        },
        # 6. 更宽的网络
        {
            'units': [128, 64],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 8,
            'name': 'batch=8,units=[128,64]'
        },
        # 7. 三层网络
        {
            'units': [96, 64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 8,
            'name': 'batch=8,3层网络'
        },
        # 8. 组合优化：低dropout+更多epochs
        {
            'units': [64, 32],
            'dropout': 0.15,
            'epochs': 150,
            'batch_size': 8,
            'name': 'batch=8,dropout=0.15,epochs=150'
        },
        # 9. 组合优化：宽网络+低dropout
        {
            'units': [96, 48],
            'dropout': 0.15,
            'epochs': 120,
            'batch_size': 8,
            'name': 'batch=8,宽网络+低dropout'
        },
        # 10. 尝试更小的batch_size
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 120,
            'batch_size': 4,
            'name': 'batch=4,极小批次'
        },
        # 11. 平衡配置
        {
            'units': [80, 40],
            'dropout': 0.18,
            'epochs': 120,
            'batch_size': 8,
            'name': 'batch=8,平衡配置'
        },
        # 12. 深度网络
        {
            'units': [128, 96, 64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 8,
            'name': 'batch=8,4层深度网络'
        }
    ]
    
    best_lstm_r2 = -float('inf')
    best_lstm_config = None
    best_lstm_results = None
    
    for i, lstm_config in enumerate(lstm_configs):
        config_name = lstm_config.pop('name')
        print(f"\n📊 测试LSTM配置 {i+1}/{len(lstm_configs)}: {config_name}")
        print(f"   参数: {lstm_config}")
        
        config = base_config.copy()
        config['lstm_config'] = lstm_config
        
        try:
            system = CarbonPricePredictionSystem(config=config)
            system.load_data('data.dta')
            system.preprocess_data()
            system.train_models()
            results, _ = system.evaluate_models()
            
            # 记录结果
            log_tuning_result(log_file, config, results, f"LSTM配置 {i+1}: {config_name}")
            
            # 检查LSTM模型性能
            if 'lstm' in results:
                lstm_r2 = results['lstm']['R²']
                print(f"   ✅ R² = {lstm_r2:.4f}, RMSE = {results['lstm']['RMSE']:.2f}")
                
                if lstm_r2 > best_lstm_r2:
                    best_lstm_r2 = lstm_r2
                    best_lstm_config = lstm_config.copy()
                    best_lstm_config['name'] = config_name
                    best_lstm_results = results['lstm'].copy()
                    print(f"   🏆 新的最佳配置！")
            else:
                print(f"   ❌ LSTM模型训练失败")
                    
        except Exception as e:
            error_msg = f"LSTM配置 {i+1} ({config_name}) 失败: {str(e)}"
            print(f"   ❌ {error_msg}")
            log_tuning_result(log_file, config, {}, error_msg)
    
    return best_lstm_config, best_lstm_results

def tune_transformer_parameters(log_file):
    """调优Transformer模型参数"""
    print("\n" + "=" * 70)
    print("🔍 开始Transformer模型参数调优")
    print("=" * 70)
    
    # 基础配置
    base_config = {
        'target_column': 'coal_price',
        'sequence_length': 60,
        'test_size': 0.2,
        'validation_size': 0.1,
        'lstm_config': {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 16
        }
    }
    
    # Transformer参数组合 - 激进简化策略，解决严重过拟合问题
    transformer_configs = [
        # 1. 极简单层配置 + 极高dropout
        {
            'd_model': 16,
            'num_heads': 2,
            'num_layers': 1,
            'dff': 64,
            'dropout': 0.6,
            'epochs': 100,
            'name': '极简单层(d=16,dropout=0.6)'
        },
        # 2. 超轻量级 + 高dropout
        {
            'd_model': 32,
            'num_heads': 2,
            'num_layers': 1,
            'dff': 128,
            'dropout': 0.5,
            'epochs': 120,
            'name': '超轻量级(d=32,dropout=0.5)'
        },
        # 3. 单层 + 更多训练
        {
            'd_model': 24,
            'num_heads': 2,
            'num_layers': 1,
            'dff': 96,
            'dropout': 0.5,
            'epochs': 150,
            'name': '单层长训练(d=24,epochs=150)'
        },
        # 4. 最小可行配置
        {
            'd_model': 16,
            'num_heads': 4,
            'num_layers': 1,
            'dff': 64,
            'dropout': 0.5,
            'epochs': 100,
            'name': '最小可行(d=16,4heads)'
        },
        # 5. 小batch_size训练
        {
            'd_model': 32,
            'num_heads': 2,
            'num_layers': 1,
            'dff': 128,
            'dropout': 0.5,
            'epochs': 100,
            'batch_size': 8,
            'name': '超轻量级+小batch(batch=8)'
        },
        # 6. 两层极简配置
        {
            'd_model': 16,
            'num_heads': 2,
            'num_layers': 2,
            'dff': 64,
            'dropout': 0.6,
            'epochs': 100,
            'name': '两层极简(d=16,dropout=0.6)'
        },
        # 7. 平衡轻量级
        {
            'd_model': 32,
            'num_heads': 4,
            'num_layers': 1,
            'dff': 128,
            'dropout': 0.4,
            'epochs': 120,
            'name': '平衡轻量级(d=32,dropout=0.4)'
        },
        # 8. 更小维度 + 更多heads
        {
            'd_model': 24,
            'num_heads': 4,
            'num_layers': 1,
            'dff': 96,
            'dropout': 0.5,
            'epochs': 100,
            'name': '小维度多heads(d=24,4heads)'
        }
    ]
    
    best_transformer_r2 = -float('inf')
    best_transformer_config = None
    best_transformer_results = None
    
    for i, transformer_config in enumerate(transformer_configs):
        config_name = transformer_config.pop('name')
        print(f"\n📊 测试Transformer配置 {i+1}/{len(transformer_configs)}: {config_name}")
        print(f"   参数: {transformer_config}")
        
        config = base_config.copy()
        config['transformer_config'] = transformer_config
        
        try:
            system = CarbonPricePredictionSystem(config=config)
            system.load_data('data.dta')
            system.preprocess_data()
            system.train_models()
            results, _ = system.evaluate_models()
            
            # 记录结果
            log_tuning_result(log_file, config, results, f"Transformer配置 {i+1}: {config_name}")
            
            # 检查Transformer模型性能
            if 'transformer' in results:
                transformer_r2 = results['transformer']['R²']
                print(f"   ✅ R² = {transformer_r2:.4f}, RMSE = {results['transformer']['RMSE']:.2f}")
                
                if transformer_r2 > best_transformer_r2:
                    best_transformer_r2 = transformer_r2
                    best_transformer_config = transformer_config.copy()
                    best_transformer_config['name'] = config_name
                    best_transformer_results = results['transformer'].copy()
                    print(f"   🏆 新的最佳配置！")
            else:
                print(f"   ❌ Transformer模型训练失败")
                    
        except Exception as e:
            error_msg = f"Transformer配置 {i+1} ({config_name}) 失败: {str(e)}"
            print(f"   ❌ {error_msg}")
            log_tuning_result(log_file, config, {}, error_msg)
    
    return best_transformer_config, best_transformer_results

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🚀 碳价格预测模型参数调优 v3.0 - 第三轮优化")
    print("=" * 70)
    print("\n📊 第二轮最佳结果:")
    print("   🏆 LSTM: R²=0.8768, RMSE=36.37 (batch_size=8)")
    print("   ❌ Transformer: R²=-0.9251 (严重过拟合)")
    print("\n🎯 第三轮目标:")
    print("   ✅ LSTM: R²>0.90, 接近RandomForest(0.943)")
    print("   ✅ Transformer: R²>0.3 (激进简化策略)")
    
    # 创建调优日志
    log_file = create_tuning_log()
    
    # 调优LSTM参数
    best_lstm_config, best_lstm_results = tune_lstm_parameters(log_file)
    
    # 调优Transformer参数
    best_transformer_config, best_transformer_results = tune_transformer_parameters(log_file)
    
    # 记录最佳配置
    print("\n" + "=" * 70)
    print("🏆 参数调优完成 - 最佳配置")
    print("=" * 70)
    
    final_log = [
        "\n" + "=" * 70,
        "第三轮调优最终结果",
        "=" * 70,
        ""
    ]
    
    if best_lstm_config and best_lstm_results:
        print(f"\n📊 最佳LSTM配置: {best_lstm_config.get('name', 'N/A')}")
        print(f"   配置: {best_lstm_config}")
        print(f"   R² = {best_lstm_results['R²']:.4f}")
        print(f"   RMSE = {best_lstm_results['RMSE']:.2f}")
        print(f"   MAE = {best_lstm_results['MAE']:.2f}")
        print(f"   MAPE = {best_lstm_results['MAPE']:.2f}%")
        
        final_log.extend([
            f"最佳LSTM配置: {best_lstm_config}",
            f"  R² = {best_lstm_results['R²']:.4f}",
            f"  RMSE = {best_lstm_results['RMSE']:.2f}",
            f"  MAE = {best_lstm_results['MAE']:.2f}",
            f"  MAPE = {best_lstm_results['MAPE']:.2f}%",
            ""
        ])
    else:
        print("\n❌ 未找到有效的LSTM配置")
        final_log.append("最佳LSTM配置: 未找到\n")
    
    if best_transformer_config and best_transformer_results:
        print(f"\n📊 最佳Transformer配置: {best_transformer_config.get('name', 'N/A')}")
        print(f"   配置: {best_transformer_config}")
        print(f"   R² = {best_transformer_results['R²']:.4f}")
        print(f"   RMSE = {best_transformer_results['RMSE']:.2f}")
        print(f"   MAE = {best_transformer_results['MAE']:.2f}")
        print(f"   MAPE = {best_transformer_results['MAPE']:.2f}%")
        
        final_log.extend([
            f"最佳Transformer配置: {best_transformer_config}",
            f"  R² = {best_transformer_results['R²']:.4f}",
            f"  RMSE = {best_transformer_results['RMSE']:.2f}",
            f"  MAE = {best_transformer_results['MAE']:.2f}",
            f"  MAPE = {best_transformer_results['MAPE']:.2f}%",
            ""
        ])
    else:
        print("\n❌ 未找到有效的Transformer配置")
        final_log.append("最佳Transformer配置: 未找到\n")
    
    final_log.extend([
        f"调优结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70
    ])
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(final_log))
    
    print("\n" + "=" * 70)
    print(f"📄 详细记录已保存至: {log_file}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
