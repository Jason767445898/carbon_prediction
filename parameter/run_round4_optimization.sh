#!/bin/bash
# 煤炭价格预测 - 第四轮参数优化执行脚本

echo "=========================================="
echo "煤炭价格预测 - 第四轮参数优化"
echo "=========================================="
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "工作目录: /Users/Jason/Desktop/code/AI/parameter"
echo "Python脚本: coal_price_optimization_round4.py"
echo ""
echo "基于第三轮最优配置 Round3_Config2_MultiHead8"
echo "  • R² = 0.358 (第二轮: 0.219, 提升 63.7%)"
echo "  • RMSE = 114.91 (第二轮: 126.74, 降低 9.3%)"
echo "  • MAPE = 9.52% (第二轮: 10.89%, 降低 12.6%)"
echo ""
echo "核心改进:"
echo "  • 8头注意力精细调优"
echo "  • 更大Attention维度 (320)"
echo "  • 正则化微调"
echo "  • LSTM容量优化"
echo "  • 学习率精调"
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
caffeinate -i python3 coal_price_optimization_round4.py

echo ""
echo "=========================================="
echo "优化任务完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
