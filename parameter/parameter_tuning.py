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

def tune_joint_parameters():
    """联合调优LSTM和Transformer模型参数"""
    print("开始联合调优LSTM和Transformer模型参数...")
    print("✨ 优化策略: 围绕第四轮成功配置微调，12轮测试\n")
    
    # 基础配置
    base_config = {
        'target_column': 'coal_price',
        'sequence_length': 60,
        'test_size': 0.2,
        'validation_size': 0.1
    }
    
    # 联合参数配置 - 第五轮优化 (12组配置，预计3小时)
    joint_configs = [
        # 配置1: 第四轮最佳配置基线重现
        {
            'lstm': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'name': '第四轮最佳配置基线'
        },
        # 配置2: LSTM+10% epochs
        {
            'lstm': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 154,  # 140*1.1
                'batch_size': 8
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'name': 'LSTM延长训练+10%'
        },
        # 配置3: LSTM-10% dropout
        {
            'lstm': {
                'units': [72, 36],
                'dropout': 0.14,  # 0.16*0.9
                'epochs': 140,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'name': 'LSTM降低dropout-10%'
        },
        # 配置4: LSTM+10% units
        {
            'lstm': {
                'units': [80, 40],  # 72*1.1≨80
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'name': 'LSTM增加神经元+10%'
        },
        # 配置5: Transformer增加d_model
        {
            'lstm': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 20,  # 16*1.25
                'num_heads': 2,
                'num_layers': 2,
                'dff': 80,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'name': 'Transformer扩容d_model+25%'
        },
        # 配置6: Transformer降低dropout
        {
            'lstm': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.55,  # 0.6-0.05
                'epochs': 100,
                'batch_size': 8
            },
            'name': 'Transformer降低dropout-0.05'
        },
        # 配置7: 联合优化-增加epochs
        {
            'lstm': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 160,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 120,
                'batch_size': 8
            },
            'name': '联合增加训练轮数'
        },
        # 配置8: 联合优化-batch_size=4
        {
            'lstm': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 4
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 4
            },
            'name': '联合极小batch_size=4'
        },
        # 配置9: LSTM深度+1层
        {
            'lstm': {
                'units': [72, 48, 24],  # 三层
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'name': 'LSTM增加深度三层'
        },
        # 配置10: Transformer深度+1层
        {
            'lstm': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 3,  # 三层
                'dff': 64,
                'dropout': 0.65,  # 增加dropout防止过拟合
                'epochs': 100,
                'batch_size': 8
            },
            'name': 'Transformer增加深度三层'
        },
        # 配置11: 精英配置-LSTM极致优化
        {
            'lstm': {
                'units': [80, 40],
                'dropout': 0.14,
                'epochs': 160,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'name': 'LSTM极致优化-多维增强'
        },
        # 配置12: 精英配置-Transformer突破
        {
            'lstm': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 8
            },
            'transformer': {
                'd_model': 24,  # 大幅增加
                'num_heads': 2,
                'num_layers': 2,
                'dff': 96,
                'dropout': 0.55,  # 降低dropout
                'epochs': 120,
                'batch_size': 8
            },
            'name': 'Transformer极致优化-突破R²0.80'
        }
    ]
    
    best_lstm_r2 = -float('inf')
    best_lstm_config = None
    best_lstm_results = None
    
    best_transformer_r2 = -float('inf')
    best_transformer_config = None
    best_transformer_results = None
    
    for i, joint_config in enumerate(joint_configs):
        config_name = joint_config['name']
        print(f"\n{'='*70}")
        print(f"🔬 测试配置 {i+1}/{len(joint_configs)}: {config_name}")
        print(f"{'='*70}")
        print(f"LSTM参数: {joint_config['lstm']}")
        print(f"Transformer参数: {joint_config['transformer']}")
        print()
        
        # 创建系统实例
        config = base_config.copy()
        config['lstm_config'] = joint_config['lstm']
        config['transformer_config'] = joint_config['transformer']
        
        try:
            system = CarbonPricePredictionSystem(config=config)
            # 使用真实数据文件
            system.load_data('data.dta')
            system.preprocess_data()
            system.train_models()
            results, _ = system.evaluate_models()
            
            # 记录结果
            log_tuning_result(config, results, f"联合配置测试 {i+1}: {config_name}")
            
            # 检查LSTM模型性能
            if 'lstm' in results:
                lstm_r2 = results['lstm']['R²']
                print(f"\n📊 LSTM结果: R²={lstm_r2:.4f}, RMSE={results['lstm']['RMSE']:.4f}")
                if lstm_r2 > best_lstm_r2:
                    best_lstm_r2 = lstm_r2
                    best_lstm_config = joint_config['lstm'].copy()
                    best_lstm_results = results['lstm'].copy()
                    print(f"   🏆 LSTM新最佳记录！")
            
            # 检查Transformer模型性能
            if 'transformer' in results:
                transformer_r2 = results['transformer']['R²']
                print(f"📊 Transformer结果: R²={transformer_r2:.4f}, RMSE={results['transformer']['RMSE']:.4f}")
                if transformer_r2 > best_transformer_r2:
                    best_transformer_r2 = transformer_r2
                    best_transformer_config = joint_config['transformer'].copy()
                    best_transformer_results = results['transformer'].copy()
                    print(f"   🏆 Transformer新最佳记录！")
                    
        except Exception as e:
            error_msg = f"联合配置测试 {i+1} ({config_name}) 失败: {str(e)}"
            print(f"\n❌ {error_msg}")
            log_tuning_result(config, {}, error_msg)
    
    return best_lstm_config, best_lstm_results, best_transformer_config, best_transformer_results

def main():
    """主函数"""
    print("🚀 开始第五轮参数调优...")
    print("基于最新运行结果 (2025-10-15 00:53):")
    print("  - RandomForest: R²=0.9290 (优秀)")
    print("  - LSTM: R²=0.574 (严重退化❌，从0.8904崩溃)")
    print("  - Transformer: R²=0.7746 (良好✅，稳定)")
    print()
    print("🎯 优化目标:")
    print("  1. LSTM恢复到R²>0.89 (重现第四轮最佳)")
    print("  2. Transformer稳定在R²>0.77 或突破至R²>0.80")
    print("  3. 终极目标: LSTM R²>0.92, Transformer R²>0.85")
    print("  4. 实验量: 12组配置 (预计3小时)")
    print()
    
    # 追加到现有日志
    with open('parameter_tuning.txt', 'a', encoding='utf-8') as f:
        f.write("\n\n" + "="*60 + "\n")
        f.write("第五轮参数调优开始\n")
        f.write("="*60 + "\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n问题诊断:\n")
        f.write("- LSTM从R²=0.8904崩溃至0.574，主配置文件可能被修改\n")
        f.write("- Transformer保持稳定(0.7874→0.7746)，简化策略有效\n")
        f.write("\n优化策略:\n")
        f.write("- LSTM: 强制恢复第四轮最佳+微调增强(12组配置)\n")
        f.write("- Transformer: 在稳定基础上微调以突破R²=0.80\n")
        f.write("- 策略: 围绕成功配置±10%微调，避免大幅跳跃\n")
        f.write("\n" + "="*60 + "\n\n")
    
    # 联合调优LSTM和Transformer参数
    print("\n" + "="*60)
    print("🔬 LSTM & Transformer 联合参数调优")
    print("="*60)
    best_lstm_config, best_lstm_results, best_transformer_config, best_transformer_results = tune_joint_parameters()
    
    # 记录最佳配置
    print("\n" + "="*60)
    print("参数调优完成 - 最佳配置")
    print("="*60)
    
    final_log = [
        "\n" + "="*60,
        "第五轮调优最终结果",
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
        "- LSTM性能恢复情况：" + (f"成功恢复(R²={best_lstm_results['R²']:.4f})" if best_lstm_results and best_lstm_results['R²'] >= 0.89 else f"进展中(R²={best_lstm_results['R²']:.4f})" if best_lstm_results else "需继续优化"),
        "- Transformer突破情况：" + (f"成功突破(R²={best_transformer_results['R²']:.4f})" if best_transformer_results and best_transformer_results['R²'] > 0.80 else f"稳定状态(R²={best_transformer_results['R²']:.4f})" if best_transformer_results else "需继续优化"),
        "- 最佳策略：围绕第四轮成功配置±10%微调",
        "- 性能评估：" + (
            "双模型均达预期" if (best_lstm_results and best_lstm_results['R²'] >= 0.89 and best_transformer_results and best_transformer_results['R²'] >= 0.77)
            else "部分达标" if (best_lstm_results and best_lstm_results['R²'] >= 0.85)
            else "需进一步优化"
        ),
        "",
        f"调优结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*60
    ]
    
    with open('parameter_tuning.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(final_log))
    
    print("\n" + "="*60)
    print("✅ 第五轮参数调优已完成！")
    print("="*60)
    if best_lstm_results:
        print(f"\n🏆 最佳LSTM模型:")
        print(f"   配置: {best_lstm_config}")
        print(f"   R² = {best_lstm_results['R²']:.4f}")
        print(f"   RMSE = {best_lstm_results['RMSE']:.4f}")
        if best_lstm_results['R²'] >= 0.89:
            print("   ✅ 已达到目标 (R²≥0.89)")
        elif best_lstm_results['R²'] >= 0.85:
            print("   ⚠️ 接近目标 (R²≥0.85)")
        else:
            print("   ❌ 未达目标，需继续优化")
    if best_transformer_results:
        print(f"\n🏆 最佳Transformer模型:")
        print(f"   配置: {best_transformer_config}")
        print(f"   R² = {best_transformer_results['R²']:.4f}")
        print(f"   RMSE = {best_transformer_results['RMSE']:.4f}")
        if best_transformer_results['R²'] >= 0.80:
            print("   ✅ 超出预期 (R²≥0.80)")
        elif best_transformer_results['R²'] >= 0.77:
            print("   ✅ 达到基本目标 (R²≥0.77)")
        else:
            print("   ⚠️ 需进一步优化")
    print("\n📄 详细记录请查看: parameter_tuning.txt")
    print("="*60)

if __name__ == "__main__":
    import platform
    
    # 在macOS上提示使用caffeinate防止休眠
    if platform.system() == 'Darwin':
        print("\n" + "="*60)
        print("⚠️  重要提示：本轮优化预计需要3小时")
        print("="*60)
        print("建议使用caffeinate防止Mac休眠：")
        print("  caffeinate -i python3 parameter_tuning.py")
        print("="*60)
        print()
    
    main()