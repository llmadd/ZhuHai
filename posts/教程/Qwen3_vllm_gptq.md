---
title: 
    zh: 'Vllm部署Qwen3-235B-A22B-GPTQ-Int4报错'
    en: 'Vllm deployment of Qwen3-235B-A22B-GPTQ-Int4 reports an error.'
date: '2025-07-21'
author: 'Hai'
coverImage: 'https://camo.githubusercontent.com/8793b3b4014d538b367ec8819dcca85e79cb8d910c808fa7849e3cd85e2ebe79/68747470733a2f2f7169616e77656e2d7265732e6f73732d616363656c65726174652d6f766572736561732e616c6979756e63732e636f6d2f6c6f676f5f7177656e332e706e67'
coverImageAlt:
    zh: 'Qwen3 图标'
    en: 'Qwen3 Icon'
tags: ['Qwen3', 'vllm', 'Qwen3-235B-A22B-GPTQ-Int4', 'size_k must divisible by BLOCK_SIZE_K']
status: 'published'
---

<!-- Chinese Content -->

# 关于使用Vllm/sglang部署Qwen3-235B-A22B-GPTQ-Int4出现报错

最近使用8卡部署Qwen3-235B-A22B-GPTQ-Int4模型出现一系列错误类似`size_k must divisible by BLOCK_SIZE_K`或者sglang一直报显存不足问题参考[MoE models error](https://github.com/vllm-project/vllm/issues/17604)。
后面发现是因为MoE模型将`tensor-parallel-size`设置为 8 tp并行过高，权重无法整除。可以参考[Qwen3 Usage Guide](https://github.com/vllm-project/vllm/issues/17327)

可以修改参数为`--tensor-parallel-size 8 --enable-expert-parallel`使用专家并行或者`--tensor-parallel-size 4 --pipeline-parallel-size 2`张量并行结合流水线并行。如果显存很大也可以结合数据并行（dp）。

## 关于使用专家并行还是张量并行结合流水线并行

我的机器是8卡A100 40GB PCIE接口 NVLINK部署Qwen3-235B-A22B-GPTQ-Int4 vllm 0.9.2版本分别对两种情况进行的压测

压测使用随机数据集输入输出均为1024tokens

数据如下：

`--tensor-parallel-size 8 --enable-expert-parallel`专家并行

单个请求

```
{
    "Time taken for tests (s)": 255.9107,
    "Number of concurrency": 1,
    "Total requests": 10,
    "Succeed requests": 10,
    "Failed requests": 0,
    "Output token throughput (tok/s)": 40.014,
    "Total token throughput (tok/s)": 81.5323,
    "Request throughput (req/s)": 0.0391,
    "Average latency (s)": 25.5892,
    "Average time to first token (s)": 0.2734,
    "Average time per output token (s)": 0.0248,
    "Average input tokens per request": 1062.5,
    "Average output tokens per request": 1024.0,
    "Average package latency (s)": 0.0248,
    "Average package per request": 1022.3
}
```

10个并发

```
{
    "Time taken for tests (s)": 78.9037,
    "Number of concurrency": 10,
    "Total requests": 20,
    "Succeed requests": 20,
    "Failed requests": 0,
    "Output token throughput (tok/s)": 259.5567,
    "Total token throughput (tok/s)": 536.8313,
    "Request throughput (req/s)": 0.2535,
    "Average latency (s)": 39.4105,
    "Average time to first token (s)": 1.1695,
    "Average time per output token (s)": 0.0374,
    "Average input tokens per request": 1093.9,
    "Average output tokens per request": 1024.0,
    "Average package latency (s)": 0.0374,
    "Average package per request": 1021.95
}
```

50个并发

```
{
    "Time taken for tests (s)": 154.7737,
    "Number of concurrency": 50,
    "Total requests": 100,
    "Succeed requests": 100,
    "Failed requests": 0,
    "Output token throughput (tok/s)": 661.6112,
    "Total token throughput (tok/s)": 1355.1853,
    "Request throughput (req/s)": 0.6461,
    "Average latency (s)": 77.1078,
    "Average time to first token (s)": 3.3566,
    "Average time per output token (s)": 0.0722,
    "Average input tokens per request": 1073.47,
    "Average output tokens per request": 1024.0,
    "Average package latency (s)": 0.0722,
    "Average package per request": 1021.96
}
```

100个并发

```
{
    "Time taken for tests (s)": 251.5154,
    "Number of concurrency": 100,
    "Total requests": 200,
    "Succeed requests": 200,
    "Failed requests": 0,
    "Output token throughput (tok/s)": 814.2641,
    "Total token throughput (tok/s)": 1676.1436,
    "Request throughput (req/s)": 0.7952,
    "Average latency (s)": 124.976,
    "Average time to first token (s)": 6.3438,
    "Average time per output token (s)": 0.1161,
    "Average input tokens per request": 1083.88,
    "Average output tokens per request": 1024.0,
    "Average package latency (s)": 0.1161,
    "Average package per request": 1021.87
}
```


`--tensor-parallel-size 4 --pipeline-parallel-size 2`张量并行结合流水线并行

单个请求

```
{
    "Time taken for tests (s)": 308.833,
    "Number of concurrency": 1,
    "Total requests": 10,
    "Succeed requests": 10,
    "Failed requests": 0,
    "Output token throughput (tok/s)": 33.1571,
    "Total token throughput (tok/s)": 67.5608,
    "Request throughput (req/s)": 0.0324,
    "Average latency (s)": 30.8813,
    "Average time to first token (s)": 0.3141,
    "Average time per output token (s)": 0.0299,
    "Average input tokens per request": 1062.5,
    "Average output tokens per request": 1024.0,
    "Average package latency (s)": 0.0299,
    "Average package per request": 1021.8
}
```

10个并发

```
{
    "Time taken for tests (s)": 75.3037,
    "Number of concurrency": 10,
    "Total requests": 20,
    "Succeed requests": 20,
    "Failed requests": 0,
    "Output token throughput (tok/s)": 271.9653,
    "Total token throughput (tok/s)": 562.4955,
    "Request throughput (req/s)": 0.2656,
    "Average latency (s)": 37.6215,
    "Average time to first token (s)": 0.9636,
    "Average time per output token (s)": 0.0359,
    "Average input tokens per request": 1093.9,
    "Average output tokens per request": 1024.0,
    "Average package latency (s)": 0.0359,
    "Average package per request": 1021.8
}
```

50个并发

```
{
    "Time taken for tests (s)": 137.0526,
    "Number of concurrency": 50,
    "Total requests": 100,
    "Succeed requests": 100,
    "Failed requests": 0,
    "Output token throughput (tok/s)": 747.1583,
    "Total token throughput (tok/s)": 1530.4123,
    "Request throughput (req/s)": 0.7296,
    "Average latency (s)": 68.3798,
    "Average time to first token (s)": 2.1127,
    "Average time per output token (s)": 0.0648,
    "Average input tokens per request": 1073.47,
    "Average output tokens per request": 1024.0,
    "Average package latency (s)": 0.0648,
    "Average package per request": 1021.93
}
```

100个并发

```
{
    "Time taken for tests (s)": 210.6894,
    "Number of concurrency": 100,
    "Total requests": 200,
    "Succeed requests": 200,
    "Failed requests": 0,
    "Output token throughput (tok/s)": 972.0469,
    "Total token throughput (tok/s)": 2000.9357,
    "Request throughput (req/s)": 0.9493,
    "Average latency (s)": 104.9446,
    "Average time to first token (s)": 3.7691,
    "Average time per output token (s)": 0.099,
    "Average input tokens per request": 1083.88,
    "Average output tokens per request": 1024.0,
    "Average package latency (s)": 0.099,
    "Average package per request": 1021.835
}
```

从数据上来看单个请求专家并行输出较快，但是在并发请求下tp结合pp有更好的表现。

根据压测结果，可以看出 TP=4 + PP=2 和 EP=8 在不同场景下各有优劣，这与MoE模型的特性以及分布式并行的设计原理密切相关。

性能差异的可能原因

(1) EP=8 单请求更快
专家并行优势：  

EP=8 将专家均匀分布在8张卡上，每个请求只需访问部分专家（如1-2个），计算和通信都集中在少数卡上，单请求的延迟更低。  

无TP的All-Reduce通信开销，前向传播更高效。  

瓶颈：  

并发请求时，不同专家可能分布在不同的卡上，导致专家负载不均衡（某些专家被频繁调用，成为热点）。  

专家间的All-to-All通信（如Token分发）在并发高时会成为瓶颈。

(2) TP=4 + PP=2 并发更强
流水线优势：  

PP=2 将模型分成两个阶段，允许异步处理多个请求（类似CPU流水线），吞吐量更高。  

TP=4 减少了单卡的显存压力，支持更大的并发batch size。  

瓶颈：  

单请求需要经过所有流水线阶段，端到端延迟较高（尤其是PP的bubble time影响）。  

TP=4 的All-Reduce通信在低并发时显得冗余，但在高并发时被均匀分摊。

<!-- English Content -->

Content in production......