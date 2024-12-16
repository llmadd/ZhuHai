---
title: "文件转表格AI小应用-File2Table"
date: "2024-12-16"
author: "Hai"
coverImage: "https://github.com/llmadd/file2table/raw/main/images/app.png"
tags: ["AI小应用", "文件转表格"]
status: "published"
---

# 文件转表格AI小应用-File2Table

一个很简单的AI小应用，借助大模型的能力结合格式化输出，将PDF、Word、txt文件提取主要数据指标转换成Excel表格文件。主要是工作中有时候需要将一些文书需要转换成表格，然后进行数据分析。

# 产品

- 产品地址：[File2Table](https://file2table.streamlit.app/)
- 源码地址：[File2Table](https://github.com/llmadd/file2table)

# 产品截图

![File2Table](https://github.com/llmadd/file2table/raw/main/images/app.png)

# 产品逻辑

读取用户文件，转为文本后进行切割，保证切割文本块大小的前提下优先按照段落、句子进行切割。
切割后文本通过大模型进行处理，将提取的数据信息转为DataFrame，然后通过Pandas进行数据处理，最后将处理后的数据保存为Excel文件。

# 提取模式 🎯

1. **数据提取模式**
   - 重要数据：仅提取关键信息
   - 详细数据：提取所有可能的数据点

2. **表格格式**
   - 仅键值对：简单的字段-数值对
   - 包含单位：添加数值单位信息
   - 包含单位和来源：完整的数据溯源信息

# 技术架构 🏗️

- 前端：Streamlit
- 数据处理：Pandas
- AI模型：支持OpenAI SDK模型
- 文件处理：
  - PDF: PyMuPDF
  - Word: python-docx/pywin32
  - TXT: 原生Python

# 后续开发计划 🗓️

- [ ] 支持更多文件格式(Image/Video)
- [ ] 支持数据分析汇图
- [ ] ......