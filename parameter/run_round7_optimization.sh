#!/bin/bash

# 煤炭价格预测 - 第七轮参数优化执行脚本
# 防止macOS休眠，确保长时间运行不中断

echo "========================================"
echo "  煤炭价格预测 - 第七轮参数优化"
echo "========================================"
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "配置数量: 12个"
echo "预计耗时: ~6-8小时"
echo ""
echo "优化策略:"
echo "  • 回归2层LSTM架构 (基于第五轮R²=0.462)"
echo "  • 探索低学习率 (第六轮发现)"
echo "  • 微调正则化和Attention维度"
echo "  • 避免单层LSTM (第六轮失败教训)"
echo ""
echo "========================================"
echo ""

# 使用caffeinate防止系统休眠（macOS）
caffeinate -i python3 /Users/Jason/Desktop/code/AI/parameter/coal_price_optimization_round7.py

echo ""
echo "========================================"
echo "第七轮优化完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
