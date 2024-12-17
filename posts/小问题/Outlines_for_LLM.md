---
title: 
    zh: '开源大模型结构化输出：输出与正则表达式匹配或遵循 JSON 架构'
    en: 'Outlines for LLM: Outputting with JSON Schema or Matching Regular Expressions'
date: "2024-12-13"
author: "Hai"
coverImage: 'https://raw.githubusercontent.com/dottxt-ai/outlines/main/docs/assets/images/logo.png'
coverImageAlt:
    zh: 'outlines项目logo'
    en: 'outlines project logo'
tags: ['outlines', '开源大模型', '大模型JSON格式输出']
status: 'published'
--- 

<!-- Chinese Content -->
# 开源大模型结构化输出：输出与正则表达式匹配或遵循 JSON 架构

结构化输出是现在AI应用中核心用例，一般大家在使用一些成熟的大模型接口时都可以通过参数去控制格式化输出Json，比如OpenAI接口中的`response_format`参数。
如何在部署自己的大模型中实现格式化输出，给大家推荐一个成熟的大模型结构化输出开源项目：[outlines](https://github.com/dottxt-ai/outlines)。
在24年8月6日，OpenAI发布了[Introducing Structured Outputs in the API](https://openai.com/index/introducing-structured-outputs-in-the-api/)，也在测试中表示，gpt-4o-2024-08-06实现了100%的结构化输出。并且在文章中鸣谢了[outlines](https://github.com/dottxt-ai/outlines)项目。

## 为什么使用outlines结构化输出

- 不会在推理过程中增加任何开销
- 支持多种输出格式，包括JSON、正则表达式、JSON Schema等
- [加快了推理速度](http://blog.dottxt.co/coalescence.html)
- [提高了基本模型 （GSM8K） 的性能](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora)
- [提高了微调模型 （CoNNL） 的性能](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora)
- [提高了模型效率（不需要提示词过多的格式化输出示例）](https://huggingface.co/blog/evaluation-structured-outputs)


## 如何使用outlines

### 安装与使用outlines

```
pip install outlines
```

初始化模型

```python
import outlines

model = outlines.models.transformers(
    "microsoft/Phi-3-mini-4k-instruct",
    device="cuda"  # 将模型部署到GPU
)
```

结合pydantic模型,格式化输出

```python
from enum import Enum
from pydantic import BaseModel, constr, conint

class Character(BaseModel):
    name: constr(max_length=10)
    age: conint(gt=18, lt=99)
    armor: (Enum('Armor', {'leather': 'leather', 'chainmail': 'chainmail', 'plate': 'plate'}))
    strength: conint(gt=1, lt=100)

generator = outlines.generate.json(model, Character)

character = generator(
    "Generate a new character for my awesome game: "
    + "name, age (between 1 and 99), armor and strength. "
    )
print(character)
# Character(name='Zara', age=25, armor=<Armor.leather: 'leather'>, strength=85)
```

## 结合VLLM框架使用outlines结构化输出

更友好的方式是我们可以结合[VLLM](https://github.com/vllm-project/vllm)大模型推理加速框架来使用outlines结构化输出并且兼容OpenAI的SDK.

示例启动VLLM服务指令：

```
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123 --guided-decoding-backend outlines
```

更详细的指令可以参考[VLLM文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)


使用OpenAI的SDK调用模型示例代码：

```python
import openai
from enum import Enum
from pydantic import BaseModel, constr, conint

class Character(BaseModel):
    name: constr(max_length=10)
    age: conint(gt=18, lt=99)
    armor: (Enum('Armor', {'leather': 'leather', 'chainmail': 'chainmail', 'plate': 'plate'}))
    strength: conint(gt=1, lt=100)

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

response = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello, world!"}],
    extra_body = {
        "guided_json": Character.model_json_schema()
    }
)
```

更多格式化输出示例

```
guided_json: Optional[Union[str, dict, BaseModel]] = Field(
    default=None,
    description=(
        "如果指定，输出将遵循 JSON 模式。"),
)

guided_regex: Optional[str] = Field(
    default=None,
    description=(
        "如果指定，输出将遵循正则表达式模式。"),
)

guided_choice: Optional[List[str]] = Field(
    default=None,
    description=(
        "如果指定，输出将是选项之一。"),
)

guided_grammar: Optional[str] = Field(
    default=None,
    description=(
        "如果指定，输出将遵循上下文无关语法。"),
)

guided_decoding_backend: Optional[str] = Field(
    default=None,
    description=(
        "如果指定，将覆盖服务器为此特定请求的默认引导解码后端。"),
)
```

<!-- English Content -->
Content in production......

