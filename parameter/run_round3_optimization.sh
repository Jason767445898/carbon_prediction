#!/bin/bash
# 煤炭价格预测 - 第三轮参数优化执行脚本

echo "=========================================="
echo "煤炭价格预测 - 第三轮参数优化"
echo "=========================================="
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "工作目录: /Users/Jason/Desktop/code/AI/parameter"
echo "Python脚本: coal_price_optimization_round3.py"
echo ""
echo "基于第二轮最优配置 Round2_Config3_BiggerAttention (R²=0.219)"
echo "核心改进:"
echo "  • 多头注意力机制 (4头/8头)"
echo "  • 更大Attention维度 (320/384)"
echo "  • 学习率精细调优"
echo "  • 增强方向感知"
echo ""
echo "配置数量: 6"
echo "预计耗时: 2.5-3小时"
echo ""
echo "=========================================="
echo ""

# 切换到工作目录
cd /Users/Jason/Desktop/code/AI/parameter

# 使用caffeinate防止系统休眠,后台运行
echo "🚀 启动优化任务 (防休眠模式)..."
caffeinate -i python3 coal_price_optimization_round3.py

echo ""
echo "=========================================="
echo "优化任务完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
