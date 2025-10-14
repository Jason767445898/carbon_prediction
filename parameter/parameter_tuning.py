#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数调优脚本
用于调整LSTM和Transformer模型的超参数
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append('/Users/Jason/Desktop/code/AI')

from carbon_price_prediction import CarbonPricePredictionSystem

def create_tuning_log():
    """创建参数调优日志文件"""
    log_content = [
        "==========================================",
        "碳价格预测模型参数调优记录",
        "==========================================",
        f"调优开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "调优目标:",
        "- 提升LSTM和Transformer模型的预测性能",
        "- 优化超参数配置",
        "- 记录调优过程和结果",
        "",
        "==========================================",
        ""
    ]
    
    with open('parameter_tuning.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_content))
    
    print("已创建参数调优日志文件: parameter_tuning.txt")

def log_tuning_result(config, results, notes=""):
    """记录调优结果"""
    log_entry = [
        f"调优时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "配置参数:",
        f"  LSTM配置: {config.get('lstm_config', 'N/A')}",
        f"  Transformer配置: {config.get('transformer_config', 'N/A')}",
        "模型性能:",
    ]
    
    for model_name, metrics in results.items():
        log_entry.append(f"  {model_name}:")
        for metric, value in metrics.items():
            if metric not in ['predictions', 'actual']:
                log_entry.append(f"    {metric}: {value:.4f}")
    
    if notes:
        log_entry.append(f"备注: {notes}")
    
    log_entry.append("-" * 50)
    log_entry.append("")
    
    with open('parameter_tuning.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_entry))
    
    print("已记录调优结果到 parameter_tuning.txt")

def tune_lstm_parameters():
    """调优LSTM模型参数"""
    print("开始调优LSTM模型参数...")
    
    # 基础配置
    base_config = {
        'target_column': 'coal_price',
        'sequence_length': 60,
        'test_size': 0.2,
        'validation_size': 0.1,
        'transformer_config': {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 4,
            'dff': 512,
            'dropout': 0.1,
            'epochs': 50
        }
    }
    
    # LSTM参数组合 - 基于batch_size=16的最佳结果继续优化
    lstm_configs = [
        # 最佳基线配置 (batch_size=16, R²=0.6778)
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 16
        },
        # 基于最佳配置：增加网络深度
        {
            'units': [128, 64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 16
        },
        # 基于最佳配置：增加网络宽度
        {
            'units': [128, 64],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 16
        },
        # 基于最佳配置：降低dropout
        {
            'units': [64, 32],
            'dropout': 0.1,
            'epochs': 100,
            'batch_size': 16
        },
        # 基于最佳配置：增加训练轮数
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 150,
            'batch_size': 16
        },
        # 基于最佳配置：减小batch_size
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 8
        },
        # 复杂网络 + 小batch_size
        {
            'units': [128, 64, 32],
            'dropout': 0.15,
            'epochs': 120,
            'batch_size': 16
        },
        # 宽网络 + 低dropout
        {
            'units': [128, 64],
            'dropout': 0.1,
            'epochs': 120,
            'batch_size': 16
        },
        # 深网络 + 正则化
        {
            'units': [96, 64, 32],
            'dropout': 0.25,
            'epochs': 100,
            'batch_size': 16
        },
        # 小batch_size + 更多epochs
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 200,
            'batch_size': 8
        }
    ]
    
    best_lstm_r2 = -float('inf')
    best_lstm_config = None
    best_lstm_results = None
    
    for i, lstm_config in enumerate(lstm_configs):
        print(f"\n测试LSTM配置 {i+1}/{len(lstm_configs)}")
        
        # 创建系统实例
        config = base_config.copy()
        config['lstm_config'] = lstm_config
        
        try:
            system = CarbonPricePredictionSystem(config=config)
            system.load_data('data.dta')
            system.preprocess_data()
            system.train_models()
            results, _ = system.evaluate_models()
            
            # 记录结果
            log_tuning_result(config, results, f"LSTM配置测试 {i+1}")
            
            # 检查LSTM模型性能
            if 'lstm' in results:
                lstm_r2 = results['lstm']['R²']
                if lstm_r2 > best_lstm_r2:
                    best_lstm_r2 = lstm_r2
                    best_lstm_config = lstm_config.copy()
                    best_lstm_results = results['lstm'].copy()
                    
        except Exception as e:
            error_msg = f"LSTM配置测试 {i+1} 失败: {str(e)}"
            print(error_msg)
            log_tuning_result(config, {}, error_msg)
    
    return best_lstm_config, best_lstm_results

def tune_transformer_parameters():
    """调优Transformer模型参数"""
    print("开始调优Transformer模型参数...")
    
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
            'batch_size': 32
        }
    }
    
    # Transformer参数组合 - 针对R²为负的问题优化
    transformer_configs = [
        # 简化模型：减少复杂度，防止过拟合
        {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 256,
            'dropout': 0.3,
            'epochs': 50
        },
        # 轻量级配置
        {
            'd_model': 32,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 128,
            'dropout': 0.3,
            'epochs': 50
        },
        # 增加正则化
        {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 3,
            'dff': 256,
            'dropout': 0.4,
            'epochs': 50
        },
        # 减少层数，增加训练轮数
        {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 256,
            'dropout': 0.2,
            'epochs': 100
        },
        # 最小配置
        {
            'd_model': 32,
            'num_heads': 2,
            'num_layers': 2,
            'dff': 128,
            'dropout': 0.3,
            'epochs': 80
        },
        # 中等复杂度 + 高dropout
        {
            'd_model': 64,
            'num_heads': 8,
            'num_layers': 2,
            'dff': 256,
            'dropout': 0.5,
            'epochs': 50
        },
        # 平衡配置
        {
            'd_model': 96,
            'num_heads': 4,
            'num_layers': 3,
            'dff': 384,
            'dropout': 0.3,
            'epochs': 60
        },
        # 超轻量级 + 更多epochs
        {
            'd_model': 32,
            'num_heads': 4,
            'num_layers': 1,
            'dff': 128,
            'dropout': 0.2,
            'epochs': 100
        }
    ]
    
    best_transformer_r2 = -float('inf')
    best_transformer_config = None
    best_transformer_results = None
    
    for i, transformer_config in enumerate(transformer_configs):
        print(f"\n测试Transformer配置 {i+1}/{len(transformer_configs)}")
        
        # 创建系统实例
        config = base_config.copy()
        config['transformer_config'] = transformer_config
        
        try:
            system = CarbonPricePredictionSystem(config=config)
            system.load_data('data.dta')
            system.preprocess_data()
            system.train_models()
            results, _ = system.evaluate_models()
            
            # 记录结果
            log_tuning_result(config, results, f"Transformer配置测试 {i+1}")
            
            # 检查Transformer模型性能
            if 'transformer' in results:
                transformer_r2 = results['transformer']['R²']
                if transformer_r2 > best_transformer_r2:
                    best_transformer_r2 = transformer_r2
                    best_transformer_config = transformer_config.copy()
                    best_transformer_results = results['transformer'].copy()
                    
        except Exception as e:
            error_msg = f"Transformer配置测试 {i+1} 失败: {str(e)}"
            print(error_msg)
            log_tuning_result(config, {}, error_msg)
    
    return best_transformer_config, best_transformer_results

def main():
    """主函数"""
    print("🚀 开始第二轮参数调优...")
    print("基于上一轮结果:")
    print("  - LSTM最佳: batch_size=16, R²=0.6778")
    print("  - Transformer问题: 所有配置R²为负，需要简化模型")
    print()
    
    # 追加到现有日志
    with open('parameter_tuning.txt', 'a', encoding='utf-8') as f:
        f.write("\n\n" + "="*60 + "\n")
        f.write("第二轮参数调优开始\n")
        f.write("="*60 + "\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n优化策略:\n")
        f.write("- LSTM: 基于batch_size=16的最佳配置继续优化\n")
        f.write("- Transformer: 简化模型结构，增加dropout防止过拟合\n")
        f.write("\n" + "="*60 + "\n\n")
    
    # 调优LSTM参数
    print("\n" + "="*60)
    print("LSTM模型参数调优")
    print("="*60)
    best_lstm_config, best_lstm_results = tune_lstm_parameters()
    
    # 调优Transformer参数
    print("\n" + "="*60)
    print("Transformer模型参数调优")
    print("="*60)
    best_transformer_config, best_transformer_results = tune_transformer_parameters()
    
    # 记录最佳配置
    print("\n" + "="*60)
    print("参数调优完成 - 最佳配置")
    print("="*60)
    
    final_log = [
        "\n" + "="*60,
        "第二轮调优最终结果",
        "="*60,
        f"最佳LSTM配置: {best_lstm_config}",
        f"  R²: {best_lstm_results['R²']:.4f}" if best_lstm_results else "  R²: N/A",
        f"  RMSE: {best_lstm_results['RMSE']:.4f}" if best_lstm_results else "  RMSE: N/A",
        f"  MAE: {best_lstm_results['MAE']:.4f}" if best_lstm_results else "  MAE: N/A",
        f"  MAPE: {best_lstm_results['MAPE']:.4f}%" if best_lstm_results else "  MAPE: N/A",
        "",
        f"最佳Transformer配置: {best_transformer_config}",
        f"  R²: {best_transformer_results['R²']:.4f}" if best_transformer_results else "  R²: N/A",
        f"  RMSE: {best_transformer_results['RMSE']:.4f}" if best_transformer_results else "  RMSE: N/A",
        f"  MAE: {best_transformer_results['MAE']:.4f}" if best_transformer_results else "  MAE: N/A",
        f"  MAPE: {best_transformer_results['MAPE']:.4f}%" if best_transformer_results else "  MAPE: N/A",
        "",
        "关键发现:",
        "- LSTM模型：小batch_size(8-16)显著提升性能",
        "- Transformer模型：需要简化结构并增加正则化",
        "- 建议：优先使用优化后的LSTM或传统机器学习模型",
        "",
        f"调优结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*60
    ]
    
    with open('parameter_tuning.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(final_log))
    
    print("\n" + "="*60)
    print("✅ 第二轮参数调优已完成！")
    print("="*60)
    if best_lstm_results:
        print(f"\n🏆 最佳LSTM模型:")
        print(f"   配置: {best_lstm_config}")
        print(f"   R² = {best_lstm_results['R²']:.4f}")
        print(f"   RMSE = {best_lstm_results['RMSE']:.4f}")
    if best_transformer_results:
        print(f"\n🏆 最佳Transformer模型:")
        print(f"   配置: {best_transformer_config}")
        print(f"   R² = {best_transformer_results['R²']:.4f}")
        print(f"   RMSE = {best_transformer_results['RMSE']:.4f}")
    print("\n📄 详细记录请查看: parameter_tuning.txt")
    print("="*60)

if __name__ == "__main__":
    main()