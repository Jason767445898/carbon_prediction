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
    
    # LSTM参数组合 - 第四轮优化配置
    lstm_configs = [
        # 组1: 基线强化 (配置1-3)
        # 配置1: 延长训练轮数
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 150,
            'batch_size': 8
        },
        # 配置2: 降低dropout，增强拟合
        {
            'units': [64, 32],
            'dropout': 0.15,
            'epochs': 150,
            'batch_size': 8
        },
        # 配置3: 进一步降低dropout
        {
            'units': [64, 32],
            'dropout': 0.10,
            'epochs': 150,
            'batch_size': 8
        },
        # 组2: 网络容量优化 (配置4-5)
        # 配置4: 增加宽度
        {
            'units': [96, 48],
            'dropout': 0.2,
            'epochs': 120,
            'batch_size': 8
        },
        # 配置5: 增加深度
        {
            'units': [96, 64, 32],
            'dropout': 0.2,
            'epochs': 120,
            'batch_size': 8
        },
        # 组3: 组合优化 (配置6-8)
        # 配置6: 宽网络+低dropout
        {
            'units': [96, 48],
            'dropout': 0.15,
            'epochs': 150,
            'batch_size': 8
        },
        # 配置7: 极小batch size
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 150,
            'batch_size': 4
        },
        # 配置8: 大网络+高正则
        {
            'units': [128, 64],
            'dropout': 0.25,
            'epochs': 120,
            'batch_size': 8
        },
        # 组4: 精细调优 (配置9-10)
        # 配置9: 在最优区间微调
        {
            'units': [80, 40],
            'dropout': 0.18,
            'epochs': 140,
            'batch_size': 8
        },
        # 配置10: 精细微调
        {
            'units': [72, 36],
            'dropout': 0.16,
            'epochs': 140,
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
    
    # Transformer参数组合 - 第四轮激进简化配置
    transformer_configs = [
        # 组1: 极简单层 (配置1-4)
        # 配置1: 最激进简化
        {
            'd_model': 16,
            'num_heads': 2,
            'num_layers': 1,
            'dff': 64,
            'dropout': 0.6,
            'epochs': 100,
            'batch_size': 8
        },
        # 配置2: 超轻量级
        {
            'd_model': 24,
            'num_heads': 2,
            'num_layers': 1,
            'dff': 96,
            'dropout': 0.5,
            'epochs': 120,
            'batch_size': 8
        },
        # 配置3: 小模型+长训练
        {
            'd_model': 32,
            'num_heads': 2,
            'num_layers': 1,
            'dff': 128,
            'dropout': 0.5,
            'epochs': 150,
            'batch_size': 8
        },
        # 配置4: 更多注意力头
        {
            'd_model': 32,
            'num_heads': 4,
            'num_layers': 1,
            'dff': 128,
            'dropout': 0.5,
            'epochs': 100,
            'batch_size': 8
        },
        # 组2: 双层探索 (配置5)
        # 配置5: 两层极简
        {
            'd_model': 16,
            'num_heads': 2,
            'num_layers': 2,
            'dff': 64,
            'dropout': 0.6,
            'epochs': 100,
            'batch_size': 8
        },
        # 组3: 平衡配置 (配置6-8)
        # 配置6: 平衡配置
        {
            'd_model': 24,
            'num_heads': 4,
            'num_layers': 1,
            'dff': 96,
            'dropout': 0.4,
            'epochs': 120,
            'batch_size': 8
        },
        # 配置7: 极小batch
        {
            'd_model': 32,
            'num_heads': 2,
            'num_layers': 1,
            'dff': 128,
            'dropout': 0.5,
            'epochs': 120,
            'batch_size': 4
        },
        # 配置8: 中等规模
        {
            'd_model': 48,
            'num_heads': 4,
            'num_layers': 1,
            'dff': 192,
            'dropout': 0.4,
            'epochs': 100,
            'batch_size': 8
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
    print("🚀 开始第四轮参数调优...")
    print("基于最新运行结果 (2025-10-14 22:45):")
    print("  - RandomForest: R²=0.9290 (优秀)")
    print("  - LSTM: R²=0.7227 (从0.8768退化，需恢复)")
    print("  - Transformer: R²=-1.2344 (严重过拟合)")
    print()
    print("🎯 优化目标:")
    print("  1. LSTM恢复到R²>0.87 (第二轮最佳水平)")
    print("  2. Transformer达到R²>0 (消除负值)")
    print("  3. 理想目标: LSTM R²>0.90, Transformer R²>0.3")
    print()
    
    # 追加到现有日志
    with open('parameter_tuning.txt', 'a', encoding='utf-8') as f:
        f.write("\n\n" + "="*60 + "\n")
        f.write("第四轮参数调优开始\n")
        f.write("="*60 + "\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n优化策略:\n")
        f.write("- LSTM: 基于第二轮成功经验(batch_size=8)，延长训练+微调dropout\n")
        f.write("- Transformer: 激进简化(d_model=16-48, 1-2层)+高dropout(0.4-0.6)\n")
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
        "第四轮调优最终结果",
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
        "- LSTM性能恢复情况：" + (f"成功恢复(R²={best_lstm_results['R²']:.4f})" if best_lstm_results and best_lstm_results['R²'] > 0.87 else "需继续优化"),
        "- Transformer过拟合解决：" + (f"已解决(R²={best_transformer_results['R²']:.4f})" if best_transformer_results and best_transformer_results['R²'] > 0 else "仍需简化"),
        "- 最佳策略：batch_size=4-8 + epochs=120-150 + dropout微调",
        "",
        f"调优结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*60
    ]
    
    with open('parameter_tuning.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(final_log))
    
    print("\n" + "="*60)
    print("✅ 第四轮参数调优已完成！")
    print("="*60)
    if best_lstm_results:
        print(f"\n🏆 最佳LSTM模型:")
        print(f"   配置: {best_lstm_config}")
        print(f"   R² = {best_lstm_results['R²']:.4f}")
        print(f"   RMSE = {best_lstm_results['RMSE']:.4f}")
        if best_lstm_results['R²'] >= 0.87:
            print("   ✅ 已达到目标 (R²≥0.87)")
        elif best_lstm_results['R²'] >= 0.85:
            print("   ⚠️ 接近目标 (R²≥0.85)")
        else:
            print("   ❌ 未达目标，需继续优化")
    if best_transformer_results:
        print(f"\n🏆 最佳Transformer模型:")
        print(f"   配置: {best_transformer_config}")
        print(f"   R² = {best_transformer_results['R²']:.4f}")
        print(f"   RMSE = {best_transformer_results['RMSE']:.4f}")
        if best_transformer_results['R²'] >= 0.3:
            print("   ✅ 超出预期 (R²≥0.3)")
        elif best_transformer_results['R²'] > 0:
            print("   ✅ 达到基本目标 (R²>0)")
        else:
            print("   ❌ 仍过拟合，需进一步简化")
    print("\n📄 详细记录请查看: parameter_tuning.txt")
    print("="*60)

if __name__ == "__main__":
    main()