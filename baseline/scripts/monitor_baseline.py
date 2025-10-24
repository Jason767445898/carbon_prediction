#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""监控Baseline训练进度"""

import os
import time
import subprocess

log_file = '/Users/Jason/Desktop/code/AI/baseline_output.log'
output_dir = '/Users/Jason/Desktop/code/AI/outputs/baseline'

print("🚀 Baseline五模型训练监控")
print("="*60)

while True:
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 显示最后50行
        print("\n最新日志（最后50行）:")
        print("-"*60)
        for line in lines[-50:]:
            print(line.rstrip())
        
        # 检查训练进度
        content = ''.join(lines)
        
        if 'RNN结果' in content:
            print("\n✅ RNN模型训练完成")
        if 'GRU结果' in content:
            print("✅ GRU模型训练完成")
        if 'LSTM结果' in content:
            print("✅ LSTM模型训练完成")
        if 'Transformer结果' in content:
            print("✅ Transformer模型训练完成")
        if 'AutoFormer结果' in content:
            print("✅ AutoFormer模型训练完成")
        
        if '程序执行完成' in content:
            print("\n🎉 所有训练完成！")
            print(f"📁 输出文件位置: {output_dir}")
            break
    
    print("\n等待中... (每30秒刷新一次)")
    time.sleep(30)

