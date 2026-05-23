# MoE All-to-All Communication Benchmark

模拟 **MoE（Mixture of Experts）架构** 在 Expert Parallel 配置下的 Token Dispatch/Combine 通信过程，对比三种 All-to-All 实现策略的性能差异。

## 为什么做这个项目

在 MoE 大模型（如 Mixtral、DeepSeek-V3）的训练和推理中，**All-to-All 通信是核心瓶颈**。Token 需要跨 GPU/节点路由到对应 Expert，通信模式直接影响整个系统的吞吐和延迟。本项目通过可量化的 Benchmark 回答：

- Naive All-to-All 的填充开销有多大？
- Bucketed（分桶）策略能节省多少带宽？
- 流水线（DeepEP 风格）在多大程度上能隐藏通信延迟？
- Token 数量、Expert 数量、Hidden Dim 如何影响通信-计算比？

## 项目结构

```
moe-comm-benchmark/
├── moe_benchmark/
│   ├── moe_layer.py            # MoE 层实现（Router + Expert FFN + Token Dispatch/Combine）
│   ├── dispatch_strategies.py  # 三种 All-to-All 策略
│   ├── benchmark.py            # Benchmark 编排与指标收集
│   ├── profiler.py             # 通信 timeline 追踪（支持 Chrome Trace 导出）
│   └── visualize.py            # 结果可视化
├── scripts/
│   ├── run_benchmark.py        # 主入口
│   └── run_all.sh              # 批量实验脚本
├── results/                    # 输出目录
└── README.md
```

## 三种策略对比

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **Naive** | 填充到统一大小后 `all_to_all` | 实现简单 | 大量填充，带宽浪费严重 |
| **Bucketed** | 按目标 rank 分桶，`all_to_all_single` 变长交换 | 零填充，带宽高效 | 需要额外的 count 交换 |
| **Pipelined** | 切分为 micro-batch，通信与计算重叠 | 隐藏通信延迟 | 需要 CUDA Stream 支持 |

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- 8GB+ 内存（单卡或多进程模拟均可）

```bash
pip install -r requirements.txt
```

### 运行

```bash
# 单策略快速测试
python scripts/run_benchmark.py --strategy bucketed

# 三策略对比
python scripts/run_benchmark.py --compare-all

# 扫遍不同 token 数量
python scripts/run_benchmark.py --sweep-tokens

# 完整实验套件
python scripts/run_benchmark.py --run-all
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hidden` | Hidden dimension | 1024 |
| `--experts` | 总 Expert 数 | 8 |
| `--top-k` | Top-K 路由 | 1 |
| `--tokens` | 每 rank token 数 | 1024 |
| `--world-size` | 进程数（rank） | 4 |
| `--backend` | `gloo` (CPU) 或 `nccl` (GPU) | gloo |
| `--iterations` | Benchmark 迭代次数 | 50 |

## 输出示例

运行 `--compare-all` 后会产生：

```
results/
├── results_latest.json              # 原始数据
├── benchmark_comparison.png         # 延迟、吞吐、通信量对比
└── latency_distribution.png         # 延迟分布直方图
```

## 简历呈现

> **MoE 通信 Benchmark 工具** | PyTorch Distributed  
> 实现 MoE 架构下 Token Dispatch/Combine 的三种 All-to-All 策略（Naive / Bucketed / Pipelined），在 EP 配置下使用多进程模拟分布式通信，对比不同 Token 数、Expert 数、Hidden Dim 下的延迟与吞吐。使用 PyTorch Profiler 完成通信-计算重叠分析，量化 All-to-All 通信占比与填充开销。
