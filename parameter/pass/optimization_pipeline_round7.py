#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第七轮参数优化 - 自动化测试流程
执行三个并行实验:
- 实验A: 增强正则化,简化架构
- 实验B: 优化学习率调度
- 实验C: 特征工程优化

创建时间: 2025-10-29 08:45
"""

import os
import shutil
import subprocess
import time
from datetime import datetime
import json

# ============================================================================
# 配置参数
# ============================================================================

WORKSPACE = "/Users/Jason/Desktop/code/AI"
MAIN_FILE = "lstm_attention_carbon_prediction.py"
BACKUP_DIR = os.path.join(WORKSPACE, "parameter")
RESULTS_FILE = os.path.join(BACKUP_DIR, "第七轮优化结果汇总.json")

# 实验配置
EXPERIMENTS = {
    "实验A_增强正则化": {
        "version": "7A",
        "description": "增强正则化,简化架构",
        "config_changes": {
            "sequence_length": 60,
            "epochs": 400,
            "batch_size": 32,
            "learning_rate": 0.00005,
            "lstm_units": 128,
            "lstm_units_2": 64,
            "lstm_units_3": 32,
            "attention_dim": 64,
            "dropout_rate": 0.5,
            "l2_reg": 0.005,
        },
        "code_changes": {
            "learning_rate_schedule": "CosineDecay",  # 保持原有
            "early_stopping_patience": 60,
            "direction_weight": 0.20,  # 保持20%不变
        }
    },
    
    "实验B_优化学习率": {
        "version": "7B",
        "description": "优化学习率调度策略",
        "config_changes": {
            "sequence_length": 90,
            "epochs": 350,
            "batch_size": 16,
            "learning_rate": 0.0001,
            "lstm_units": 256,
            "lstm_units_2": 128,
            "lstm_units_3": 64,
            "attention_dim": 128,
            "dropout_rate": 0.4,
            "l2_reg": 0.001,
        },
        "code_changes": {
            "learning_rate_schedule": "ExponentialDecay",  # 改为指数衰减
            "add_reduce_lr": True,  # 添加ReduceLROnPlateau
            "early_stopping_patience": 50,
            "direction_weight": 0.20,  # 保持20%不变
        }
    },
    
    "实验C_特征工程": {
        "version": "7C",
        "description": "特征工程优化,降低price_lag_1主导",
        "config_changes": {
            "sequence_length": 90,
            "epochs": 300,
            "batch_size": 16,
            "learning_rate": 0.0001,
            "lstm_units": 256,
            "lstm_units_2": 128,
            "lstm_units_3": 64,
            "attention_dim": 128,
            "dropout_rate": 0.4,
            "l2_reg": 0.001,
        },
        "code_changes": {
            "learning_rate_schedule": "CosineDecay",
            "early_stopping_patience": 50,
            "direction_weight": 0.20,  # 保持20%不变
            "feature_engineering": True,  # 启用特征工程优化
        }
    }
}

# ============================================================================
# 辅助函数
# ============================================================================

def log_message(message, level="INFO"):
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def backup_current_code(version):
    """备份当前代码"""
    backup_file = os.path.join(BACKUP_DIR, f"优化版本{version}_代码备份.py")
    source_file = os.path.join(WORKSPACE, MAIN_FILE)
    
    shutil.copy2(source_file, backup_file)
    log_message(f"✅ 代码已备份至: {backup_file}")
    return backup_file

def modify_config_params(config_changes):
    """修改CONFIG参数"""
    file_path = os.path.join(WORKSPACE, MAIN_FILE)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换CONFIG字典中的参数
    for key, value in config_changes.items():
        # 查找并替换配置项
        if isinstance(value, str):
            pattern = f"'{key}': [^,\\n]+"
            replacement = f"'{key}': '{value}'"
        else:
            pattern = f"'{key}': [^,\\n]+"
            replacement = f"'{key}': {value}"
        
        import re
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    log_message(f"✅ CONFIG参数已更新")

def modify_learning_rate_schedule(schedule_type):
    """修改学习率调度策略"""
    file_path = os.path.join(WORKSPACE, MAIN_FILE)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找学习率调度代码段
    for i, line in enumerate(lines):
        if "lr_schedule = tf.keras.optimizers.schedules.CosineDecay" in line:
            if schedule_type == "ExponentialDecay":
                # 替换为指数衰减
                lines[i] = "    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(\n"
                lines[i+1] = "        initial_learning_rate=CONFIG['learning_rate'],\n"
                lines[i+2] = "        decay_steps=1000,\n"
                lines[i+3] = "        decay_rate=0.96,\n"
                lines[i+4] = "        staircase=True\n"
                lines[i+5] = "    )\n"
            break
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    log_message(f"✅ 学习率调度已改为: {schedule_type}")

def add_reduce_lr_callback():
    """添加ReduceLROnPlateau回调"""
    file_path = os.path.join(WORKSPACE, MAIN_FILE)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找callbacks定义
    import re
    
    # 在EarlyStopping后添加ReduceLROnPlateau
    pattern = r"(callbacks = \[.*?EarlyStopping\([^\]]+\),)"
    replacement = r"\1\n    ReduceLROnPlateau(\n        monitor='val_loss',\n        patience=30,\n        factor=0.7,\n        min_lr=1e-7,\n        verbose=1\n    ),"
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    log_message(f"✅ 已添加ReduceLROnPlateau回调")

def modify_direction_weight(weight):
    """修改方向损失权重"""
    file_path = os.path.join(WORKSPACE, MAIN_FILE)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换损失函数中的权重
    import re
    pattern = r"total_loss = (0\.\d+) \* huber \+ (0\.\d+) \* direction_component"
    huber_weight = weight
    direction_weight = 1.0 - weight
    replacement = f"total_loss = {huber_weight} * huber + {direction_weight} * direction_component  # 第七轮优化: 保持20%方向权重"
    
    content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    log_message(f"✅ 方向损失权重已设置为: {direction_weight*100}%")

def add_feature_engineering():
    """添加特征工程优化代码"""
    file_path = os.path.join(WORKSPACE, MAIN_FILE)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 在create_sequences之前添加特征工程代码
    feature_eng_code = """
    # 特征工程优化: 降低price_lag主导性
    price_lag_cols = [col for col in data.columns if 'price_lag' in col]
    for col in price_lag_cols:
        if col in data.columns:
            data[col] = data[col] * 0.5  # 降权50%
    
    # 增强技术指标信号
    tech_indicators = ['rsi', 'volatility', 'momentum', 'ma_5', 'ma_10', 'ema_12', 'ema_26']
    for col in tech_indicators:
        if col in data.columns:
            data[col] = data[col] * 1.5  # 增强50%
    
    print(f"   • 应用特征工程优化: price_lag降权50%, 技术指标增强50%")
"""
    
    # 查找合适的插入位置
    for i, line in enumerate(lines):
        if "X_sequences, y_sequences = create_sequences" in line:
            lines.insert(i, feature_eng_code)
            break
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    log_message(f"✅ 已添加特征工程优化代码")

def run_training():
    """执行训练"""
    log_message("🚀 开始训练...")
    
    # 使用caffeinate防止休眠
    cmd = f"caffeinate -i python3 {os.path.join(WORKSPACE, MAIN_FILE)}"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        if result.returncode == 0:
            log_message("✅ 训练完成")
            return True, result.stdout
        else:
            log_message(f"❌ 训练失败: {result.stderr}", level="ERROR")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        log_message("❌ 训练超时(2小时)", level="ERROR")
        return False, "Timeout"
    except Exception as e:
        log_message(f"❌ 训练异常: {str(e)}", level="ERROR")
        return False, str(e)

def extract_results_from_log():
    """从最新的日志文件中提取结果"""
    logs_dir = os.path.join(WORKSPACE, "outputs/logs")
    
    # 查找最新的日志文件
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('_analysis_report.txt')]
    if not log_files:
        return None
    
    latest_log = sorted(log_files)[-1]
    log_path = os.path.join(logs_dir, latest_log)
    
    results = {}
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 提取关键指标
        import re
        
        r2_match = re.search(r'R².*?:\s*(-?\d+\.\d+)', content)
        if r2_match:
            results['R²'] = float(r2_match.group(1))
        
        mape_match = re.search(r'MAPE.*?:\s*(\d+\.\d+)%', content)
        if mape_match:
            results['MAPE'] = float(mape_match.group(1))
        
        direction_match = re.search(r'方向准确率.*?:\s*(\d+\.\d+)%', content)
        if direction_match:
            results['方向准确率'] = float(direction_match.group(1))
        
        mae_match = re.search(r'MAE.*?:\s*(\d+\.\d+)', content)
        if mae_match:
            results['MAE'] = float(mae_match.group(1))
        
        rmse_match = re.search(r'RMSE.*?:\s*(\d+\.\d+)', content)
        if rmse_match:
            results['RMSE'] = float(rmse_match.group(1))
    
    results['log_file'] = latest_log
    
    return results

def restore_backup(backup_file):
    """恢复备份"""
    target_file = os.path.join(WORKSPACE, MAIN_FILE)
    shutil.copy2(backup_file, target_file)
    log_message(f"✅ 已从备份恢复: {backup_file}")

# ============================================================================
# 主流程
# ============================================================================

def run_experiment(exp_name, exp_config):
    """执行单个实验"""
    log_message("="*80)
    log_message(f"🔬 开始实验: {exp_name}")
    log_message(f"   描述: {exp_config['description']}")
    log_message("="*80)
    
    version = exp_config['version']
    
    # 1. 备份当前代码
    backup_file = backup_current_code(version)
    
    try:
        # 2. 修改CONFIG参数
        modify_config_params(exp_config['config_changes'])
        
        # 3. 应用代码修改
        code_changes = exp_config['code_changes']
        
        # 修改学习率调度
        if 'learning_rate_schedule' in code_changes:
            if code_changes['learning_rate_schedule'] == "ExponentialDecay":
                modify_learning_rate_schedule("ExponentialDecay")
        
        # 添加ReduceLROnPlateau
        if code_changes.get('add_reduce_lr'):
            add_reduce_lr_callback()
        
        # 修改方向权重
        modify_direction_weight(0.80)  # 80% Huber + 20% 方向
        
        # 应用特征工程
        if code_changes.get('feature_engineering'):
            add_feature_engineering()
        
        # 4. 执行训练
        success, output = run_training()
        
        if not success:
            log_message(f"❌ {exp_name} 训练失败", level="ERROR")
            return {
                'experiment': exp_name,
                'version': version,
                'status': 'FAILED',
                'error': output,
                'timestamp': datetime.now().isoformat()
            }
        
        # 5. 提取结果
        results = extract_results_from_log()
        
        if results:
            log_message(f"✅ {exp_name} 完成!")
            log_message(f"   R²: {results.get('R²', 'N/A')}")
            log_message(f"   MAPE: {results.get('MAPE', 'N/A')}%")
            log_message(f"   方向准确率: {results.get('方向准确率', 'N/A')}%")
            
            return {
                'experiment': exp_name,
                'version': version,
                'status': 'SUCCESS',
                'results': results,
                'config': exp_config,
                'timestamp': datetime.now().isoformat()
            }
        else:
            log_message(f"⚠️ {exp_name} 无法提取结果", level="WARNING")
            return {
                'experiment': exp_name,
                'version': version,
                'status': 'NO_RESULTS',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        log_message(f"❌ {exp_name} 异常: {str(e)}", level="ERROR")
        return {
            'experiment': exp_name,
            'version': version,
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    
    finally:
        # 恢复原始代码(从第一次备份)
        # 注意: 这里需要保留最后一个实验的代码,所以注释掉
        # restore_backup(backup_file)
        pass

def main():
    """主函数"""
    log_message("🚀 第七轮参数优化 - 自动化测试流程启动")
    log_message(f"   工作目录: {WORKSPACE}")
    log_message(f"   实验数量: {len(EXPERIMENTS)}")
    log_message("")
    
    # 先备份初始版本(优化版本6)
    initial_backup = backup_current_code("6_initial")
    
    all_results = []
    
    # 依次执行三个实验
    for exp_name, exp_config in EXPERIMENTS.items():
        # 每次实验前恢复到初始版本
        restore_backup(initial_backup)
        
        result = run_experiment(exp_name, exp_config)
        all_results.append(result)
        
        # 休息10秒
        log_message("⏸️  休息10秒后继续下一个实验...")
        time.sleep(10)
    
    # 保存所有结果
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    log_message("="*80)
    log_message("🎉 所有实验完成!")
    log_message(f"   结果已保存至: {RESULTS_FILE}")
    log_message("="*80)
    
    # 打印汇总
    print("\n" + "="*80)
    print("📊 实验结果汇总")
    print("="*80)
    
    for result in all_results:
        print(f"\n实验: {result['experiment']}")
        print(f"版本: {result['version']}")
        print(f"状态: {result['status']}")
        
        if result['status'] == 'SUCCESS' and 'results' in result:
            res = result['results']
            print(f"  R²: {res.get('R²', 'N/A')}")
            print(f"  MAPE: {res.get('MAPE', 'N/A')}%")
            print(f"  方向准确率: {res.get('方向准确率', 'N/A')}%")
        elif result['status'] == 'FAILED':
            print(f"  错误: {result.get('error', 'Unknown')}")
