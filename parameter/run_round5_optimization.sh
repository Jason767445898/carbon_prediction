#!/bin/bash
# 煤炭价格预测 - 第五轮参数优化执行脚本

echo "=========================================="
echo "煤炭价格预测 - 第五轮参数优化"
echo "=========================================="
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "工作目录: /Users/Jason/Desktop/code/AI/parameter"
echo "Python脚本: coal_price_optimization_round5.py"
echo ""
echo "基于第四轮失败经验,回归稳定架构"
echo "  第二轮最优: R² = 0.219 (单头Attention)"
echo "  第三轮最优: R² = 0.358 (8头,但不稳定)"
echo "  第四轮: 全部失败 (R² < 0)"
echo ""
echo "核心改进:"
echo "  • 简化LSTM层数 (3层→2层)"
echo "  • 调整序列长度 (45/60/75)"
echo "  • 优化batch size (16/32)"
echo "  • 增加训练轮数 (600)"
echo "  • 回归稳定的单头Attention"
echo ""
echo "配置数量: 6"
echo "预计耗时: 2.5-3小时"
echo ""
echo "=========================================="
echo ""

# 切换到工作目录
cd /Users/Jason/Desktop/code/AI/parameter

# 使用caffeinate防止系统休眠
echo "🚀 启动优化任务 (防休眠模式)..."
caffeinate -i python3 coal_price_optimization_round5.py

echo ""
echo "=========================================="
echo "优化任务完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
