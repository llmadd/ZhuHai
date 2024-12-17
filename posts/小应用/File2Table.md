---
title:
  zh: "File2Table - 文件转表格工具"
  en: "File2Table - File to Table Converter"
date: "2024-03-19"
author: "Hai"
coverImage: "https://github.com/llmadd/file2table/raw/main/images/app.png"
coverImageAlt:
  zh: "File2Table工具界面截图"
  en: "File2Table Interface Screenshot"
tags: ["python", "streamlit", "pandas", "tools"]
status: "published"
---

<!-- Chinese Content -->
# File2Table - 文件转表格工具

File2Table 是一个简单的文件转换工具，可以将各种格式的文件转换为表格形式。借助大模型的能力结合格式化输出，将PDF、Word、txt文件提取主要数据指标转换成Excel表格文件。

# 产品链接

- 产品地址：[File2Table](https://file2table.streamlit.app/)
- 源码地址：[GitHub](https://github.com/llmadd/file2table)

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


<!-- English Content -->
# File2Table - File to Table Converter

File2Table is a simple conversion tool that transforms various file formats into tabular data. Leveraging AI capabilities with formatted output, it extracts key data indicators from PDF, Word, and txt files, converting them into Excel spreadsheets.

# Product Links

- Product URL: [File2Table](https://file2table.streamlit.app/)
- Source Code: [GitHub](https://github.com/llmadd/file2table)

# How It Works

The tool reads user files and splits them into text segments, prioritizing paragraph and sentence boundaries while maintaining appropriate chunk sizes. The text is then processed through an AI model, extracting data into a DataFrame. Finally, Pandas handles the data processing before saving to Excel format.

# Extraction Modes 🎯

1. **Data Extraction Mode**
   - Essential Data: Extracts only key information
   - Detailed Data: Extracts all possible data points

2. **Table Format**
   - Key-Value Only: Simple field-value pairs
   - With Units: Includes measurement units
   - Complete Info: Full data provenance

# Technical Stack 🏗️

- Frontend: Streamlit
- Data Processing: Pandas
- AI Model: OpenAI SDK models
- File Processing:
  - PDF: PyMuPDF
  - Word: python-docx/pywin32
  - TXT: Native Python

# Future Development 🗓️

- [ ] Support for more file formats (Image/Video)
- [ ] Data analysis and visualization
- [ ] ......