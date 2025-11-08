#!/bin/bash
# 简单煤炭价格预测运行脚本（防休眠）

echo "========================================="
echo "简单单层 LSTM + Attention 煤炭价格预测"
echo "========================================="
echo ""
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 使用 caffeinate 防止系统休眠
caffeinate -i python3 simple_lstm_attention_coal.py

echo ""
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
