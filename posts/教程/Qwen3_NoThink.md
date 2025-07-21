---
title: 
    zh: 'Qwen3模型修改默认不思考'
    en: 'SQLModel Tutorial'
date: '2025-07-21'
author: 'Hai'
coverImage: 'https://camo.githubusercontent.com/8793b3b4014d538b367ec8819dcca85e79cb8d910c808fa7849e3cd85e2ebe79/68747470733a2f2f7169616e77656e2d7265732e6f73732d616363656c65726174652d6f766572736561732e616c6979756e63732e636f6d2f6c6f676f5f7177656e332e706e67'
coverImageAlt:
    zh: 'Qwen3 图标'
    en: 'Qwen3 Icon'
tags: ['Qwen3', 'vllm', default no think']
status: 'published'
---

<!-- Chinese Content -->

# 使用Vllm 部署 Qwen3设置默认不思考

修改模型文件`tokenizer_config.json`中`chat_template`参数改为
```
 "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking %}\n    {% else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
```
或者`vllm serve Qwen/Qwen3 --chat-template ./qwen3_defaultnonthinking.jinja`

将下面jinja语法复制保存为`qwen3_defaultnonthinking.jinja`文件 

```
{% if tools %}
    {{ '<|im_start|>system\n' }}
    {% if messages[0].role == 'system' %}
        {{ messages[0].content + '\n\n' }}
    {% endif %}
    {{ "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {% for tool in tools %}
        {{ "\n" }}
        {{ tool | tojson }}
    {% endfor %}
    {{ "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{% else %}
    {% if messages[0].role == 'system' %}
        {{ '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {% endif %}
{% endif %}

{% set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{% for message in messages[::-1] %}
    {% set index = (messages|length - 1) - loop.index0 %}
    {% if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {% set ns.multi_step_tool = false %}
        {% set ns.last_query_index = index %}
    {% endif %}
{% endfor %}

{% for message in messages %}
    {% if message.content is string %}
        {% set content = message.content %}
    {% else %}
        {% set content = '' %}
    {% endif %}

    {% if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{ '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {% elif message.role == "assistant" %}
        {% set reasoning_content = '' %}
        {% if message.reasoning_content is string %}
            {% set reasoning_content = message.reasoning_content %}
        {% else %}
            {% if '</think>' in content %}
                {% set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {% set content = content.split('</think>')[-1].lstrip('\n') %}
            {% endif %}
        {% endif %}

        {% if loop.index0 > ns.last_query_index %}
            {% if loop.last or (not loop.last and reasoning_content) %}
                {{ '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {% else %}
                {{ '<|im_start|>' + message.role + '\n' + content }}
            {% endif %}
        {% else %}
            {{ '<|im_start|>' + message.role + '\n' + content }}
        {% endif %}

        {% if message.tool_calls %}
            {% for tool_call in message.tool_calls %}
                {% if (loop.first and content) or (not loop.first) %}
                    {{ '\n' }}
                {% endif %}
                {% if tool_call.function %}
                    {% set tool_call = tool_call.function %}
                {% endif %}
                {{ '<tool_call>\n{"name": "' }}
                {{ tool_call.name }}
                {{ '", "arguments": ' }}
                {% if tool_call.arguments is string %}
                    {{ tool_call.arguments }}
                {% else %}
                    {{ tool_call.arguments | tojson }}
                {% endif %}
                {{ '}\n</tool_call>' }}
            {% endfor %}
        {% endif %}
        {{ '<|im_end|>\n' }}
    {% elif message.role == "tool" %}
        {% if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{ '<|im_start|>user' }}
        {% endif %}
        {{ '\n<tool_response>\n' }}
        {{ content }}
        {{ '\n</tool_response>' }}
        {% if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{ '<|im_end|>\n' }}
        {% endif %}
    {% endif %}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
    {% if enable_thinking is defined and enable_thinking %}
    {% else %}
        {{ '<think>\n\n</think>\n\n' }}
    {% endif %}
{% endif %}
```


要完全禁用思考 可以将下面jinja语法保存为 `qwen3_nonthinking.jinja`文件并在启动时制定 `vllm serve Qwen/Qwen3 --chat-template ./qwen3_nonthinking.jinja`。
参考[qwen3教程](https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html)

```
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}
{%- endif %}
```


<!-- English Content -->

Content in production......