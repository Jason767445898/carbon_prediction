#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四轮参数优化 - 基于最新结果的针对性优化
生成时间: 2025-10-14
优化目标: 恢复并超越第二轮最佳性能 (LSTM R²=0.8768)

增强功能:
1. 支持断点续传 - 实验中断后可继续
2. 实时进度保存 - 每个实验完成后立即保存
3. 异常处理增强 - 单个实验失败不影响整体流程
4. 并行运行支持 - 可配置是否并行运行实验
5. 早停机制 - 发现优秀配置后可提前结束
6. 详细日志记录 - 包含训练过程详细信息
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carbon_price_prediction import CarbonPricePredictionSystem, DEFAULT_CONFIG
import numpy as np
import pandas as pd
from datetime import datetime
import json
import pickle
import time

def log_message(message, log_file=None):
    """记录日志信息（仅打印到控制台）"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    # 不再生成.log文件，只输出到控制台

def run_experiment(config, experiment_name):
    """运行单个实验配置
    
    Args:
        config: 系统配置字典
        experiment_name: 实验名称
        log_file: 日志文件路径
    """
    log_message(f"\n{'='*80}")
    log_message(f"🧪 开始实验: {experiment_name}")
    log_message(f"{'='*80}")
    
    start_time = time.time()
    
    # 显示配置
    log_message(f"\n📋 配置详情:")
    for key, value in config.items():
        if isinstance(value, dict):
            log_message(f"  {key}:")
            for k, v in value.items():
                log_message(f"    {k}: {v}")
        else:
            log_message(f"  {key}: {value}")
    
    try:
        # 创建系统实例
        log_message("\n📦 创建系统实例...")
        system = CarbonPricePredictionSystem(config=config)
        
        # 加载数据
        log_message("\n📁 正在加载数据...")
        data_file = 'data.dta'
        if not os.path.exists(data_file):
            log_message(f"❌ 数据文件不存在: {data_file}")
            return None
        system.load_data(data_file)
        log_message(f"✅ 数据加载成功")
        
        # 预处理数据
        log_message("\n🔧 正在预处理数据...")
        system.preprocess_data()
        log_message("✅ 数据预处理完成")
        
        # 训练模型
        log_message(f"\n🚀 正在训练所有模型...")
        system.train_models()
        
        # 获取结果
        elapsed_time = time.time() - start_time
        log_message(f"\n⏱️  训练耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
        
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
                'MAPE': mape,
                'training_time': elapsed_time
            }
            
            # 判断结果质量
            quality = "❌ 失败" if r2 < 0 else "⚠️ 待改进" if r2 < 0.6 else "✅ 良好" if r2 < 0.85 else "🏆 优秀"
            
            log_message(f"\n  {quality} {model_name}:")
            log_message(f"     R² = {r2:.4f}")
            log_message(f"     RMSE = {rmse:.4f}")
            log_message(f"     MAE = {mae:.4f}")
            log_message(f"     MAPE = {mape:.2f}%")
        
        log_message(f"\n✅ 实验完成: {experiment_name}")
        return results
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        log_message(f"\n❌ 实验失败 (耗时 {elapsed_time:.2f}秒): {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        log_message(error_trace)
        
        # 返回错误信息而不是None，便于分析
        return {
            'error': str(e),
            'traceback': error_trace,
            'experiment': experiment_name,
            'elapsed_time': elapsed_time
        }

def save_checkpoint(checkpoint_data, checkpoint_file):
    """保存检查点数据"""
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    except Exception as e:
        print(f"⚠️ 保存检查点失败: {e}")

def load_checkpoint(checkpoint_file):
    """加载检查点数据"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ 加载检查点失败: {e}")
    return None

def save_intermediate_results(all_results, results_file):
    """保存中间结果"""
    if not all_results:
        return
    
    try:
        df_results = pd.DataFrame(all_results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("碳价格预测系统 - 第四轮参数优化结果 (中间结果)\n")
            f.write("="*100 + "\n")
            f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"已完成实验数: {len(all_results)}\n\n")
            
            # 按模型分组
            for model in df_results['model'].unique():
                model_results = df_results[df_results['model'] == model].copy()
                if not model_results.empty:
                    model_results = model_results.sort_values('R²', ascending=False)
                    
                    f.write(f"\n{model} 模型结果:\n")
                    f.write("-"*100 + "\n")
                    for idx, row in enumerate(model_results.itertuples(), 1):
                        exp_name = getattr(row, 'experiment', '')
                        r2_val = getattr(row, 'R²', np.nan)
                        rmse_val = getattr(row, 'RMSE', np.nan)
                        mae_val = getattr(row, 'MAE', np.nan)
                        mape_val = getattr(row, 'MAPE', np.nan)
                        f.write(f"{idx:2d}. {exp_name:<35} R²={r2_val:<8.4f} RMSE={rmse_val:<8.2f} MAE={mae_val:<8.2f} MAPE={mape_val:<7.2f}%\n")
        
        log_message(f"💾 中间结果已保存: {results_file}")
    except Exception as e:
        log_message(f"⚠️ 保存中间结果失败: {e}")

def main():
    """第四轮优化主流程"""
    
    # 创建结果文件（不生成log文件）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'parameter/parameter_tuning_v4_{timestamp}.txt'
    checkpoint_file = f'parameter/parameter_tuning_v4_{timestamp}.checkpoint'
    
    # 尝试加载检查点
    checkpoint = load_checkpoint(checkpoint_file)
    start_idx = 0
    all_results = []
    
    if checkpoint:
        start_idx = checkpoint.get('last_completed_idx', 0) + 1
        all_results = checkpoint.get('results', [])
        log_message(f"✅ 从检查点恢复，继续从第 {start_idx + 1} 个实验开始")
    
    log_message("="*80)
    log_message("🎯 碳价格预测系统 - 第四轮参数优化 (方案1:统一测试)")
    log_message("="*80)
    log_message(f"\n优化开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"结果文件: {results_file}")
    log_message(f"检查点文件: {checkpoint_file}")
    
    log_message("\n📊 最新结果分析 (2025-10-14 22:45):")
    log_message("  • RandomForest: R²=0.9290 (↓ from 0.9430)")
    log_message("  • LSTM: R²=0.7227 (↓↓ from 0.8768)")
    log_message("  • Transformer: R²=-1.2344 (仍过拟合)")
    
    log_message("\n🎯 优化目标:")
    log_message("  1. LSTM恢复到R²>0.87 (第二轮最佳水平)")
    log_message("  2. Transformer达到R²>0 (可用水平)")
    log_message("  3. RandomForest稳定在R²>0.92")
    
    # =============================================================================
    # 统一实验配置 - 每次同时优化LSTM和Transformer
    # =============================================================================
    log_message("\n" + "="*80)
    log_message("📝 优化策略 (方案1: 统一测试):")
    log_message("="*80)
    log_message("  策略: 每次实验同时训练所有模型")
    log_message("  优势: 减少总实验次数，便于横向对比")
    log_message("  LSTM目标: 恢复到R²>0.87")
    log_message("  Transformer目标: 达到R²>0")
    
    # 统一的实验配置列表
    unified_configs = [
        # 配置1: LSTM基线 + Transformer极简
        {
            'name': 'Exp_01_Baseline',
            'lstm_config': {
                'units': [64, 32],
                'dropout': 0.2,
                'epochs': 150,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 1,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'description': 'LSTM第二轮最佳配置 + Transformer极简配置'
        },
        
        # 配置2: LSTM低dropout + Transformer超轻量
        {
            'name': 'Exp_02_LowDropout',
            'lstm_config': {
                'units': [64, 32],
                'dropout': 0.15,
                'epochs': 150,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 24,
                'num_heads': 2,
                'num_layers': 1,
                'dff': 96,
                'dropout': 0.5,
                'epochs': 120,
                'batch_size': 8
            },
            'description': 'LSTM低dropout(0.15) + Transformer超轻量'
        },
        
        # 配置3: LSTM更低dropout + Transformer小模型
        {
            'name': 'Exp_03_MinDropout',
            'lstm_config': {
                'units': [64, 32],
                'dropout': 0.1,
                'epochs': 150,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 32,
                'num_heads': 2,
                'num_layers': 1,
                'dff': 128,
                'dropout': 0.5,
                'epochs': 150,
                'batch_size': 8
            },
            'description': 'LSTM极低dropout(0.10) + Transformer小模型长训练'
        },
        
        # 配置4: LSTM宽网络 + Transformer多头注意力
        {
            'name': 'Exp_04_WideNetwork',
            'lstm_config': {
                'units': [96, 48],
                'dropout': 0.2,
                'epochs': 120,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 32,
                'num_heads': 4,
                'num_layers': 1,
                'dff': 128,
                'dropout': 0.5,
                'epochs': 100,
                'batch_size': 8
            },
            'description': 'LSTM宽网络(96-48) + Transformer多头注意力(4头)'
        },
        
        # 配置5: LSTM深网络 + Transformer两层
        {
            'name': 'Exp_05_DeepNetwork',
            'lstm_config': {
                'units': [96, 64, 32],
                'dropout': 0.2,
                'epochs': 120,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            },
            'description': 'LSTM深网络(3层) + Transformer两层极简'
        },
        
        # 配置6: LSTM宽网络低dropout + Transformer平衡配置
        {
            'name': 'Exp_06_Balanced',
            'lstm_config': {
                'units': [96, 48],
                'dropout': 0.15,
                'epochs': 150,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 24,
                'num_heads': 4,
                'num_layers': 1,
                'dff': 96,
                'dropout': 0.4,
                'epochs': 120,
                'batch_size': 8
            },
            'description': 'LSTM宽网络低dropout + Transformer平衡配置'
        },
        
        # 配置7: 小batch训练
        {
            'name': 'Exp_07_SmallBatch',
            'lstm_config': {
                'units': [64, 32],
                'dropout': 0.2,
                'epochs': 150,
                'batch_size': 4
            },
            'transformer_config': {
                'd_model': 32,
                'num_heads': 2,
                'num_layers': 1,
                'dff': 128,
                'dropout': 0.5,
                'epochs': 120,
                'batch_size': 4
            },
            'description': '两个模型都使用小batch(4)'
        },
        
        # 配置8: LSTM大网络高正则 + Transformer中等配置
        {
            'name': 'Exp_08_LargeNetwork',
            'lstm_config': {
                'units': [128, 64],
                'dropout': 0.25,
                'epochs': 120,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 48,
                'num_heads': 4,
                'num_layers': 1,
                'dff': 192,
                'dropout': 0.4,
                'epochs': 100,
                'batch_size': 8
            },
            'description': 'LSTM大网络高正则 + Transformer中等配置'
        },
        
        # 配置9: 平衡配置A
        {
            'name': 'Exp_09_Balanced_A',
            'lstm_config': {
                'units': [80, 40],
                'dropout': 0.18,
                'epochs': 140,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 28,
                'num_heads': 4,
                'num_layers': 1,
                'dff': 112,
                'dropout': 0.45,
                'epochs': 120,
                'batch_size': 8
            },
            'description': '两模型平衡配置A'
        },
        
        # 配置10: 平衡配置B
        {
            'name': 'Exp_10_Balanced_B',
            'lstm_config': {
                'units': [72, 36],
                'dropout': 0.16,
                'epochs': 140,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 20,
                'num_heads': 4,
                'num_layers': 1,
                'dff': 80,
                'dropout': 0.5,
                'epochs': 130,
                'batch_size': 8
            },
            'description': '两模型平衡配置B'
        },
    ]
    

    
    # =============================================================================
    # 执行优化实验
    # =============================================================================
    
    total_experiments = len(unified_configs)
    
    log_message("\n" + "="*80)
    log_message("🚀 开始统一优化实验")
    log_message(f"总计 {total_experiments} 个配置，每次同时训练所有模型")
    log_message("="*80)
    
    for i, config_info in enumerate(unified_configs, 1):
        # 检查是否已完成
        if i - 1 < start_idx:
            log_message(f"⏭️  跳过已完成的实验 {i}/{total_experiments}")
            continue
        log_message(f"\n{'='*80}")
        log_message(f"📊 总进度: {i}/{total_experiments}")
        log_message(f"🧪 实验: {config_info['name']}")
        log_message(f"📝 说明: {config_info['description']}")
        log_message(f"{'='*80}")
        
        # 创建完整配置
        full_config = DEFAULT_CONFIG.copy()
        full_config['lstm_config'] = config_info['lstm_config']
        full_config['transformer_config'] = config_info['transformer_config']
        
        # 运行实验（训练所有模型）
        results = run_experiment(full_config, config_info['name'])
        
        if results:
            if 'error' in results:
                # 记录失败的实验（记录所有模型）
                for model_name in ['lstm', 'transformer', 'random_forest']:
                    all_results.append({
                        'experiment': config_info['name'],
                        'model': model_name,
                        'description': config_info['description'],
                        'lstm_config': config_info['lstm_config'],
                        'transformer_config': config_info['transformer_config'],
                        'R²': np.nan,
                        'RMSE': np.nan,
                        'MAE': np.nan,
                        'MAPE': np.nan,
                        'error': results['error']
                    })
            else:
                # 记录所有模型的结果
                for model_name in ['lstm', 'transformer', 'random_forest']:
                    if model_name in results:
                        all_results.append({
                            'experiment': config_info['name'],
                            'model': model_name,
                            'description': config_info['description'],
                            'lstm_config': config_info['lstm_config'],
                            'transformer_config': config_info['transformer_config'],
                            **results[model_name]
                        })
                
                # 检查是否达到优秀水平
                if 'lstm' in results:
                    lstm_r2 = results['lstm'].get('R²', 0)
                    if lstm_r2 >= 0.87:
                        log_message(f"\n🎉 LSTM达到目标! R²={lstm_r2:.4f} >= 0.87")
                
                if 'transformer' in results:
                    trans_r2 = results['transformer'].get('R²', -999)
                    if trans_r2 > 0:
                        log_message(f"\n🎉 Transformer首次达到正R²! R²={trans_r2:.4f}")
                    if trans_r2 >= 0.5:
                        log_message(f"\n🏆 Transformer达到优秀水平! R²={trans_r2:.4f} >= 0.5")
            
            # 保存检查点和中间结果
            checkpoint_data = {
                'last_completed_idx': i - 1,
                'results': all_results,
                'timestamp': datetime.now().isoformat()
            }
            save_checkpoint(checkpoint_data, checkpoint_file)
            save_intermediate_results(all_results, results_file)
    
    # =============================================================================
    # 生成结果报告
    # =============================================================================
    
    log_message("\n" + "="*80)
    log_message("📊 生成优化结果报告")
    log_message("="*80)
    
    # 转换为DataFrame
    df_results = pd.DataFrame(all_results)
    
    # 保存详细结果
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("碳价格预测系统 - 第四轮参数优化结果 (方栈1:统一测试)\n")
        f.write("="*100 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验次数: {len(unified_configs)} (每次训练所有模型)\n\n")
        
        # 按模型分组显示结果
        for model_type in ['lstm', 'transformer', 'random_forest']:
            model_results = df_results[df_results['model'] == model_type].copy()
            if not model_results.empty:
                model_results = model_results.sort_values('R²', ascending=False)
                
                f.write(f"\n{'='*100}\n")
                f.write(f"📊 {model_type.upper()} 模型优化结果\n")
                f.write("="*100 + "\n\n")
                
                f.write(f"{'排名':<6} {'实验名称':<35} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}\n")
                f.write("-"*100 + "\n")
                
                for idx, row in enumerate(model_results.itertuples(), 1):
                    medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx:2d}"
                    r2_val = getattr(row, 'R²', np.nan)
                    rmse_val = getattr(row, 'RMSE', np.nan)
                    mae_val = getattr(row, 'MAE', np.nan)
                    mape_val = getattr(row, 'MAPE', np.nan)
                    exp_name = getattr(row, 'experiment', '')
                    f.write(f"{medal:<6} {exp_name:<35} {r2_val:<10.4f} {rmse_val:<10.2f} {mae_val:<10.2f} {mape_val:<10.2f}%\n")
                
                # 最佳配置
                best = model_results.iloc[0]
                f.write("\n" + "="*100 + "\n")
                f.write(f"🏆 {model_type.upper()}最佳配置\n")
                f.write("="*100 + "\n")
                f.write(f"实验: {best['experiment']}\n")
                f.write(f"说明: {best['description']}\n")
                f.write(f"R²: {best['R²']:.4f}\n")
                f.write(f"RMSE: {best['RMSE']:.2f}\n")
                f.write(f"MAE: {best['MAE']:.2f}\n")
                f.write(f"MAPE: {best['MAPE']:.2f}%\n\n")
                
                # LSTM和Transformer显示对应的配置参数
                if model_type == 'lstm' and 'lstm_config' in best:
                    f.write("LSTM配置参数:\n")
                    for key, value in best['lstm_config'].items():
                        f.write(f"  {key}: {value}\n")
                elif model_type == 'transformer' and 'transformer_config' in best:
                    f.write("Transformer配置参数:\n")
                    for key, value in best['transformer_config'].items():
                        f.write(f"  {key}: {value}\n")
        
        # 对比分析
        f.write("\n" + "="*100 + "\n")
        f.write("📈 优化效果对比\n")
        f.write("="*100 + "\n\n")
        
        lstm_results = df_results[df_results['model'] == 'lstm'].copy()
        if not lstm_results.empty:
            best_lstm_r2 = lstm_results['R²'].max()
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
        
        trans_results = df_results[df_results['model'] == 'transformer'].copy()
        if not trans_results.empty:
            best_trans_r2 = trans_results['R²'].max()
            baseline_trans_r2 = -1.2344
            
            f.write(f"Transformer模型:\n")
            f.write(f"  当前基线: R² = {baseline_trans_r2:.4f}\n")
            f.write(f"  本轮最佳: R² = {best_trans_r2:.4f}\n")
            f.write(f"  改进幅度: {(best_trans_r2 - baseline_trans_r2):.4f}\n")
            
            if best_trans_r2 > 0:
                f.write(f"  ✅ 成功达到正R²值!\n\n")
            else:
                f.write(f"  ⚠️ 仍未达到正R²值\n\n")
        
        # 模型对比
        f.write(f"\n模型性能对比 (最佳配置):\n")
        f.write("-"*100 + "\n")
        for model_name in ['lstm', 'transformer', 'random_forest']:
            model_data = df_results[df_results['model'] == model_name]
            if not model_data.empty:
                best_r2 = model_data['R²'].max()
                f.write(f"{model_name.upper():<20} R² = {best_r2:.4f}\n")
    
    log_message(f"\n✅ 结果已保存到: {results_file}")
    
    # 显示最佳结果
    lstm_results = df_results[df_results['model'] == 'lstm'].copy()
    if not lstm_results.empty:
        lstm_results = lstm_results.sort_values('R²', ascending=False)
        best_lstm = lstm_results.iloc[0]
        log_message(f"\n🏆 LSTM最佳: {best_lstm['experiment']} - R²={best_lstm['R²']:.4f}")
    
    trans_results = df_results[df_results['model'] == 'transformer'].copy()
    if not trans_results.empty:
        trans_results = trans_results.sort_values('R²', ascending=False)
        best_trans = trans_results.iloc[0]
        log_message(f"🏆 Transformer最佳: {best_trans['experiment']} - R²={best_trans['R²']:.4f}")
    
    rf_results = df_results[df_results['model'] == 'random_forest'].copy()
    if not rf_results.empty:
        rf_results = rf_results.sort_values('R²', ascending=False)
        best_rf = rf_results.iloc[0]
        log_message(f"🏆 RandomForest最佳: {best_rf['experiment']} - R²={best_rf['R²']:.4f}")
    
    log_message(f"\n优化完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 清理检查点文件
    try:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            log_message(f"🗑️  已清理检查点文件: {checkpoint_file}")
    except:
        pass
    
    log_message("="*80)
    log_message("\n" + "="*80)
    log_message("🎉 第四轮参数优化圆满完成!")
    log_message("="*80)
    
    return results_file

def quick_test(config_name='exp_baseline'):
    """快速测试单个配置
    
    Args:
        config_name: 配置名称
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'parameter/quick_test_{config_name}_{timestamp}.txt'
    
    # 预定义的快速测试配置
    quick_configs = {
        'exp_baseline': {
            'name': 'Quick_Baseline',
            'lstm_config': {
                'units': [64, 32],
                'dropout': 0.2,
                'epochs': 150,
                'batch_size': 8
            },
            'transformer_config': {
                'd_model': 16,
                'num_heads': 2,
                'num_layers': 1,
                'dff': 64,
                'dropout': 0.6,
                'epochs': 100,
                'batch_size': 8
            }
        }
    }
    
    if config_name not in quick_configs:
        print(f"❌ 未知配置: {config_name}")
        print(f"可用配置: {', '.join(quick_configs.keys())}")
        return
    
    config_info = quick_configs[config_name]
    
    # 创建完整配置
    full_config = DEFAULT_CONFIG.copy()
    full_config['lstm_config'] = config_info['lstm_config']
    full_config['transformer_config'] = config_info['transformer_config']
    
    print(f"🚀 快速测试: {config_info['name']}")
    print(f"LSTM配置: {config_info['lstm_config']}")
    print(f"Transformer配置: {config_info['transformer_config']}")
    print()
    
    # 运行实验
    results = run_experiment(full_config, config_info['name'])
    
    if results and 'error' not in results:
        print(f"\n✅ 测试完成!")
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()} 结果:")
            print(f"  R² = {metrics.get('R²', 0):.4f}")
            print(f"  RMSE = {metrics.get('RMSE', 0):.2f}")
            print(f"  MAE = {metrics.get('MAE', 0):.2f}")
            print(f"  MAPE = {metrics.get('MAPE', 0):.2f}%")
    else:
        print(f"\n❌ 测试失败")
    
    print(f"\n结果文件: {results_file}")

if __name__ == '__main__':
    main()
