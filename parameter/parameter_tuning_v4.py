#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四轮参数优化 - 基于最新结果的针对性优化
生成时间: 2025-10-14
优化目标: 恢复并超越第二轮最佳性能 (LSTM R²=0.8768)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carbon_price_prediction import CarbonPricePredictionSystem, DEFAULT_CONFIG
import numpy as np
import pandas as pd
from datetime import datetime

def log_message(message, log_file=None):
    """记录日志信息"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

def run_experiment(config, experiment_name, log_file):
    """运行单个实验配置"""
    log_message(f"\n{'='*80}", log_file)
    log_message(f"🧪 开始实验: {experiment_name}", log_file)
    log_message(f"{'='*80}", log_file)
    
    # 显示配置
    log_message(f"\n📋 配置详情:", log_file)
    for key, value in config.items():
        if isinstance(value, dict):
            log_message(f"  {key}:", log_file)
            for k, v in value.items():
                log_message(f"    {k}: {v}", log_file)
        else:
            log_message(f"  {key}: {value}", log_file)
    
    try:
        # 创建系统实例
        system = CarbonPricePredictionSystem(config=config)
        
        # 加载数据
        log_message("\n📁 正在加载数据...", log_file)
        system.load_data('data.dta')
        
        # 预处理数据
        log_message("🔧 正在预处理数据...", log_file)
        system.preprocess_data()
        
        # 训练模型
        log_message("🚀 正在训练模型...", log_file)
        system.train_models()
        
        # 获取结果
        results = {}
        for model_name, metrics in system.predictions.items():
            r2 = metrics.get('r2', np.nan)
            rmse = metrics.get('rmse', np.nan)
            mae = metrics.get('mae', np.nan)
            mape = metrics.get('mape', np.nan)
            
            results[model_name] = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
            
            log_message(f"\n  ✅ {model_name}:", log_file)
            log_message(f"     R² = {r2:.4f}", log_file)
            log_message(f"     RMSE = {rmse:.4f}", log_file)
            log_message(f"     MAE = {mae:.4f}", log_file)
            log_message(f"     MAPE = {mape:.2f}%", log_file)
        
        return results
        
    except Exception as e:
        log_message(f"\n❌ 实验失败: {str(e)}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        return None

def main():
    """第四轮优化主流程"""
    
    # 创建日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'parameter/parameter_tuning_v4_{timestamp}.log'
    results_file = f'parameter/parameter_tuning_v4_{timestamp}.txt'
    
    log_message("="*80, log_file)
    log_message("🎯 碳价格预测系统 - 第四轮参数优化", log_file)
    log_message("="*80, log_file)
    log_message(f"\n优化开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    
    log_message("\n📊 最新结果分析 (2025-10-14 22:45):", log_file)
    log_message("  • RandomForest: R²=0.9290 (↓ from 0.9430)", log_file)
    log_message("  • LSTM: R²=0.7227 (↓↓ from 0.8768)", log_file)
    log_message("  • Transformer: R²=-1.2344 (仍过拟合)", log_file)
    
    log_message("\n🎯 优化目标:", log_file)
    log_message("  1. LSTM恢复到R²>0.87 (第二轮最佳水平)", log_file)
    log_message("  2. Transformer达到R²>0 (可用水平)", log_file)
    log_message("  3. RandomForest稳定在R²>0.92", log_file)
    
    # =============================================================================
    # LSTM优化配置 - 聚焦最佳参数区间
    # =============================================================================
    log_message("\n" + "="*80, log_file)
    log_message("📝 LSTM优化策略:", log_file)
    log_message("="*80, log_file)
    log_message("  基于第二轮最佳配置: batch_size=8, units=[64,32], dropout=0.2", log_file)
    log_message("  当前退化原因分析: 配置未改变，可能是随机性或数据问题", log_file)
    log_message("  优化方向: 增强训练稳定性，微调关键参数", log_file)
    
    lstm_configs = [
        # 配置1: 验证第二轮最佳配置（增加epochs确保充分训练）
        {
            'name': 'LSTM_01_Baseline_Extended',
            'config': {
                'units': [64, 32],
                'dropout': 0.2,
                'epochs': 150,  # 增加训练轮数
                'batch_size': 8
            },
            'description': '第二轮最佳配置+延长训练'
        },
        
        # 配置2-3: 降低dropout，提高拟合能力
        {
            'name': 'LSTM_02_Lower_Dropout_015',
            'config': {
                'units': [64, 32],
                'dropout': 0.15,
                'epochs': 150,
                'batch_size': 8
            },
            'description': '降低dropout到0.15'
        },
        {
            'name': 'LSTM_03_Lower_Dropout_010',
            'config': {
                'units': [64, 32],
                'dropout': 0.1,
                'epochs': 150,
                'batch_size': 8
            },
            'description': '降低dropout到0.10'
        },
        
        # 配置4-5: 适度增加网络容量
        {
            'name': 'LSTM_04_Wider_Network',
            'config': {
                'units': [96, 48],
                'dropout': 0.2,
                'epochs': 120,
                'batch_size': 8
            },
            'description': '增加网络宽度'
        },
        {
            'name': 'LSTM_05_Deeper_Network',
            'config': {
                'units': [96, 64, 32],
                'dropout': 0.2,
                'epochs': 120,
                'batch_size': 8
            },
            'description': '增加网络深度'
        },
        
        # 配置6: 组合优化 - 宽网络+低dropout
        {
            'name': 'LSTM_06_Wide_LowDrop',
            'config': {
                'units': [96, 48],
                'dropout': 0.15,
                'epochs': 150,
                'batch_size': 8
            },
            'description': '宽网络+低dropout'
        },
        
        # 配置7: 测试更小batch_size
        {
            'name': 'LSTM_07_Smaller_Batch',
            'config': {
                'units': [64, 32],
                'dropout': 0.2,
                'epochs': 150,
                'batch_size': 4
            },
            'description': '更小的batch size'
        },
        
        # 配置8: 大网络+高正则
        {
            'name': 'LSTM_08_Large_HighReg',
            'config': {
                'units': [128, 64],
                'dropout': 0.25,
                'epochs': 120,
                'batch_size': 8
            },
            'description': '大网络+高正则化'
        },
        
        # 配置9-10: 平衡配置
        {
            'name': 'LSTM_09_Balanced_A',
            'config': {
                'units': [80, 40],
                'dropout': 0.18,
                'epochs': 140,
                'batch_size': 8
            },
            'description': '平衡配置A'
        },
        {
            'name': 'LSTM_10_Balanced_B',
            'config': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 8
            },
            'description': '平衡配置B'
        },
    ]
    
    # =============================================================================
    # Transformer优化配置 - 极简化防止过拟合
    # =============================================================================
    log_message("\n" + "="*80, log_file)
    log_message("📝 Transformer优化策略:", log_file)
    log_message("="*80, log_file)
    log_message("  问题: 持续严重过拟合 (R²<0)", log_file)
    log_message("  策略: 激进简化模型，极高正则化", log_file)
    log_message("  目标: 首先达到R²>0，再逐步提升", log_file)
    
    transformer_configs = [
        # 配置1: 极简单层 - 最小可行Transformer
        {
            'name': 'Transformer_01_Minimal',
            'config': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 1,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'description': '极简配置(参数量~5K)'
        },
        
        # 配置2: 超轻量级
        {
            'name': 'Transformer_02_UltraLight',
            'config': {
                'd_model': 24,
                'num_heads': 2,
                'num_layers': 1,
                'dff': 96,
                'dropout': 0.5,
                'epochs': 120,
                'batch_size': 8
            },
            'description': '超轻量级配置'
        },
        
        # 配置3: 小模型+长训练
        {
            'name': 'Transformer_03_Small_LongTrain',
            'config': {
                'd_model': 32,
                'num_heads': 2,
                'num_layers': 1,
                'dff': 128,
                'dropout': 0.5,
                'epochs': 150,
                'batch_size': 8
            },
            'description': '小模型+延长训练'
        },
        
        # 配置4: 增加注意力头
        {
            'name': 'Transformer_04_MoreHeads',
            'config': {
                'd_model': 32,
                'num_heads': 4,
                'num_layers': 1,
                'dff': 128,
                'dropout': 0.5,
                'epochs': 100,
                'batch_size': 8
            },
            'description': '更多注意力头'
        },
        
        # 配置5: 两层极简
        {
            'name': 'Transformer_05_TwoLayer_Mini',
            'config': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'description': '两层极简配置'
        },
        
        # 配置6: 平衡配置
        {
            'name': 'Transformer_06_Balanced',
            'config': {
                'd_model': 24,
                'num_heads': 4,
                'num_layers': 1,
                'dff': 96,
                'dropout': 0.4,
                'epochs': 120,
                'batch_size': 8
            },
            'description': '平衡配置'
        },
        
        # 配置7: 更小batch
        {
            'name': 'Transformer_07_TinyBatch',
            'config': {
                'd_model': 32,
                'num_heads': 2,
                'num_layers': 1,
                'dff': 128,
                'dropout': 0.5,
                'epochs': 120,
                'batch_size': 4
            },
            'description': '极小batch size'
        },
        
        # 配置8: 中等配置
        {
            'name': 'Transformer_08_Medium',
            'config': {
                'd_model': 48,
                'num_heads': 4,
                'num_layers': 1,
                'dff': 192,
                'dropout': 0.4,
                'epochs': 100,
                'batch_size': 8
            },
            'description': '中等规模配置'
        },
    ]
    
    # =============================================================================
    # 执行优化实验
    # =============================================================================
    
    all_results = []
    
    # LSTM实验
    log_message("\n" + "="*80, log_file)
    log_message("🚀 开始LSTM模型优化实验", log_file)
    log_message("="*80, log_file)
    
    for i, config_info in enumerate(lstm_configs, 1):
        log_message(f"\n{'='*80}", log_file)
        log_message(f"进度: LSTM {i}/{len(lstm_configs)} - {config_info['name']}", log_file)
        log_message(f"说明: {config_info['description']}", log_file)
        log_message(f"{'='*80}", log_file)
        
        # 创建完整配置
        full_config = DEFAULT_CONFIG.copy()
        full_config['lstm_config'] = config_info['config']
        
        # 运行实验
        results = run_experiment(full_config, config_info['name'], log_file)
        
        if results and 'lstm' in results:
            all_results.append({
                'experiment': config_info['name'],
                'model': 'LSTM',
                'description': config_info['description'],
                'config': config_info['config'],
                **results['lstm']
            })
    
    # Transformer实验
    log_message("\n" + "="*80, log_file)
    log_message("🚀 开始Transformer模型优化实验", log_file)
    log_message("="*80, log_file)
    
    for i, config_info in enumerate(transformer_configs, 1):
        log_message(f"\n{'='*80}", log_file)
        log_message(f"进度: Transformer {i}/{len(transformer_configs)} - {config_info['name']}", log_file)
        log_message(f"说明: {config_info['description']}", log_file)
        log_message(f"{'='*80}", log_file)
        
        # 创建完整配置
        full_config = DEFAULT_CONFIG.copy()
        full_config['transformer_config'] = config_info['config']
        
        # 运行实验
        results = run_experiment(full_config, config_info['name'], log_file)
        
        if results and 'transformer' in results:
            all_results.append({
                'experiment': config_info['name'],
                'model': 'Transformer',
                'description': config_info['description'],
                'config': config_info['config'],
                **results['transformer']
            })
    
    # =============================================================================
    # 生成结果报告
    # =============================================================================
    
    log_message("\n" + "="*80, log_file)
    log_message("📊 生成优化结果报告", log_file)
    log_message("="*80, log_file)
    
    # 转换为DataFrame
    df_results = pd.DataFrame(all_results)
    
    # 保存详细结果
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("碳价格预测系统 - 第四轮参数优化结果\n")
        f.write("="*100 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # LSTM结果
        f.write("\n" + "="*100 + "\n")
        f.write("📊 LSTM模型优化结果\n")
        f.write("="*100 + "\n\n")
        
        lstm_results = df_results[df_results['model'] == 'LSTM'].copy()
        if not lstm_results.empty:
            lstm_results = lstm_results.sort_values('R²', ascending=False)
            
            f.write(f"{'排名':<6} {'实验名称':<30} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}\n")
            f.write("-"*100 + "\n")
            
            for idx, row in enumerate(lstm_results.itertuples(), 1):
                medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx:2d}"
                f.write(f"{medal:<6} {row.experiment:<30} {row._4:<10.4f} {row.RMSE:<10.2f} {row.MAE:<10.2f} {row.MAPE:<10.2f}%\n")
            
            # 最佳配置
            best = lstm_results.iloc[0]
            f.write("\n" + "="*100 + "\n")
            f.write("🏆 LSTM最佳配置\n")
            f.write("="*100 + "\n")
            f.write(f"实验: {best['experiment']}\n")
            f.write(f"说明: {best['description']}\n")
            f.write(f"R²: {best['R²']:.4f}\n")
            f.write(f"RMSE: {best['RMSE']:.2f}\n")
            f.write(f"MAE: {best['MAE']:.2f}\n")
            f.write(f"MAPE: {best['MAPE']:.2f}%\n\n")
            f.write("配置参数:\n")
            for key, value in best['config'].items():
                f.write(f"  {key}: {value}\n")
        
        # Transformer结果
        f.write("\n" + "="*100 + "\n")
        f.write("📊 Transformer模型优化结果\n")
        f.write("="*100 + "\n\n")
        
        trans_results = df_results[df_results['model'] == 'Transformer'].copy()
        if not trans_results.empty:
            trans_results = trans_results.sort_values('R²', ascending=False)
            
            f.write(f"{'排名':<6} {'实验名称':<30} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}\n")
            f.write("-"*100 + "\n")
            
            for idx, row in enumerate(trans_results.itertuples(), 1):
                medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx:2d}"
                f.write(f"{medal:<6} {row.experiment:<30} {row._4:<10.4f} {row.RMSE:<10.2f} {row.MAE:<10.2f} {row.MAPE:<10.2f}%\n")
            
            # 最佳配置
            best = trans_results.iloc[0]
            f.write("\n" + "="*100 + "\n")
            f.write("🏆 Transformer最佳配置\n")
            f.write("="*100 + "\n")
            f.write(f"实验: {best['experiment']}\n")
            f.write(f"说明: {best['description']}\n")
            f.write(f"R²: {best['R²']:.4f}\n")
            f.write(f"RMSE: {best['RMSE']:.2f}\n")
            f.write(f"MAE: {best['MAE']:.2f}\n")
            f.write(f"MAPE: {best['MAPE']:.2f}%\n\n")
            f.write("配置参数:\n")
            for key, value in best['config'].items():
                f.write(f"  {key}: {value}\n")
        
        # 对比分析
        f.write("\n" + "="*100 + "\n")
        f.write("📈 优化效果对比\n")
        f.write("="*100 + "\n\n")
        
        if not lstm_results.empty:
            best_lstm_r2 = lstm_results.iloc[0]['R²']
            baseline_lstm_r2 = 0.7227  # 最新运行结果
            target_lstm_r2 = 0.8768    # 第二轮最佳
            
            f.write(f"LSTM模型:\n")
            f.write(f"  当前基线: R² = {baseline_lstm_r2:.4f}\n")
            f.write(f"  第二轮最佳: R² = {target_lstm_r2:.4f}\n")
            f.write(f"  本轮最佳: R² = {best_lstm_r2:.4f}\n")
            f.write(f"  改进幅度: {(best_lstm_r2 - baseline_lstm_r2):.4f}\n")
            
            if best_lstm_r2 >= target_lstm_r2:
                f.write(f"  ✅ 达到或超越第二轮最佳水平!\n\n")
            else:
                f.write(f"  ⚠️ 未达到第二轮水平，差距: {(target_lstm_r2 - best_lstm_r2):.4f}\n\n")
        
        if not trans_results.empty:
            best_trans_r2 = trans_results.iloc[0]['R²']
            baseline_trans_r2 = -1.2344
            
            f.write(f"Transformer模型:\n")
            f.write(f"  当前基线: R² = {baseline_trans_r2:.4f}\n")
            f.write(f"  本轮最佳: R² = {best_trans_r2:.4f}\n")
            f.write(f"  改进幅度: {(best_trans_r2 - baseline_trans_r2):.4f}\n")
            
            if best_trans_r2 > 0:
                f.write(f"  ✅ 成功达到正R²值!\n\n")
            else:
                f.write(f"  ⚠️ 仍未达到正R²值\n\n")
    
    log_message(f"\n✅ 结果已保存到: {results_file}", log_file)
    log_message(f"✅ 日志已保存到: {log_file}", log_file)
    
    # 显示最佳结果
    if not lstm_results.empty:
        best_lstm = lstm_results.iloc[0]
        log_message(f"\n🏆 LSTM最佳: {best_lstm['experiment']} - R²={best_lstm['R²']:.4f}", log_file)
    
    if not trans_results.empty:
        best_trans = trans_results.iloc[0]
        log_message(f"🏆 Transformer最佳: {best_trans['experiment']} - R²={best_trans['R²']:.4f}", log_file)
    
    log_message(f"\n优化完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_message("="*80, log_file)

if __name__ == '__main__':
    main()
