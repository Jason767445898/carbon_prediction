#!/bin/bash
# 煤炭价格预测 - 第六轮优化执行脚本
# 使用 caffeinate 防止 macOS 休眠

echo "=================================="
echo "  煤炭价格预测 - 第六轮优化"
echo "  策略: 探索单层LSTM架构"
echo "=================================="
echo ""

# 防止系统休眠执行优化
caffeinate -i python3 /Users/Jason/Desktop/code/AI/parameter/coal_price_optimization_round6.py

echo ""
echo "=================================="
echo "  第六轮优化执行完成!"
echo "=================================="
