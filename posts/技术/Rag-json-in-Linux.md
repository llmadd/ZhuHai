---
title: 'RAG中大模型的幻觉有可能是Json字符编码的问题'
date: '2024-12-05'
author: 'Hai'
coverImage: '/paper.png'
tags: ['RAG', 'json', 'Linux']
status: 'published'
---

# RAG中大模型的幻觉有可能是Json字符编码的问题

今天遇到一个很有趣的事情，在基于RAG做一个Excel绘制图片的工具，我在Windows上测试的时候，发现绘制出来的图片是正常的，但是到了Linux上，绘制出来的图片就出现了问题,经常出现一些错误的地区，我最初以为是模型幻觉引起的，在DeBug后发现将excel使用pandas读取后，再转换为json，在format给prompt时字符转义为ASCII编码，从而导致了图片中很多中文地区被模型错误输出。

这个问题主要与操作系统的默认编码设置和 Python 的处理方式有关。让我解释一下：

1. **Windows vs Linux 的默认编码差异**：
```python
# Windows 通常默认使用
PYTHONIOENCODING=cp936 (简体中文 Windows)
或 utf-8

# Linux 通常默认使用
PYTHONIOENCODING=ascii
或 LANG=C
```

2. **数据处理流程中的编码转换**：
```python
# 在 Linux 上
df.to_json(...) -> Unicode字符串 
                -> 系统默认用 ASCII 处理
                -> \u 转义序列

# 在 Windows 上
df.to_json(...) -> Unicode字符串 
                -> 系统默认用 UTF-8/CP936 处理
                -> 保持中文显示
```

3. **解决方案的原理**：

错误代码：
```python
df_json = current_df.to_json(orient="split", force_ascii=False)
# 使用 json.dumps获取str
json_str = json.dumps(json.loads(df_json), ensure_ascii=False)
```
在Windows上，由于默认编码为UTF-8，所以不会出现这个问题，但是在Linux上，由于默认编码为ASCII，所以会出现这个问题。

修正后代码：
```python
# 完整的解码-编码过程
df_json = current_df.to_json(orient="split", force_ascii=False)
# 第一步：json.loads() 将字符串解析成 Python 对象，这步会正确解析 Unicode 转义序列
parsed_json = json.loads(df_json)
# 第二步：json.dumps() 重新将 Python 对象转换为字符串，通过 ensure_ascii=False 确保使用原始字符
json_str = json.dumps(parsed_json, ensure_ascii=False)
```

要永久解决这个问题，你可以：

1. 设置环境变量：
```bash
# 在 Linux 服务器的环境变量中添加
export PYTHONIOENCODING=utf-8
export LANG=zh_CN.UTF-8
```

2. 在代码开头设置默认编码：
```python
import sys
import locale

# 设置默认编码
sys.stdout.reconfigure(encoding='utf-8')
locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
```

3. 或者在读取 Excel 时就确保编码：
```python
def read_excel_with_encoding(file_path):
    df = pd.read_excel(file_path, sheet_name=0, index_col=0)
    # 确保列名和索引是 UTF-8 编码
    df.columns = df.columns.map(lambda x: str(x))
    df.index = df.index.map(lambda x: str(x))
    return df
```

这就是为什么你的解决方案有效 - 它显式地处理了编码转换过程，不依赖系统默认设置。
