# 魔搭平台大语言模型部署与对比测试项目



## 1.1 项目概述

本项目旨在通过魔搭平台（ModelScope）部署和对比测试三个主流大语言模型，包括：
- **通义千问Qwen-7B-Chat**
- **智谱ChatGLM3-6B** 
- **百川2-7B-对话模型**

通过标准化的测试案例，对这些模型的语言理解能力、对话质量和推理性能进行横向对比分析。

## 1.2 环境准备

### 1. 注册并登录魔搭平台

1. 访问魔搭平台官网：https://modelscope.cn/
2. 点击右上角"登录"按钮
3. 选择"阿里云账号登录"
   
### 2. 创建Notebook环境

1. 在魔搭平台主页，点击"创建空间"
2.  点击"创建"并等待环境启动

## 2.1 Qwen-7B-Chat 部署流程
> 模型来源：Qwen 团队 / ModelScope\
> 项目主页：[https://www.modelscope.cn/qwen/Qwen-7B-Chat](https://www.modelscope.cn/qwen/Qwen-7B-Chat)

### 1. 安装 Miniconda（如系统没有）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
export PATH="/opt/conda/bin:$PATH"
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 创建环境并激活

```bash
conda create -n qwen_env python=3.10 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen_env
```
### 3. 安装依赖（CPU）

```bash
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -U pip setuptools wheel
pip install \
  intel-extension-for-transformers==1.4.2 \
  neural-compressor==2.5 \
  transformers==4.33.3 \
  modelscope==1.9.5 \
  pydantic==1.10.13 \
  sentencepiece \
  tiktoken \
  einops \
  transformers_stream_generator \
  uvicorn \
  fastapi \
  yacs \
  setuptools_scm
pip install fschat --use-pep517
pip install tqdm huggingface-hub
```
### 4.下载模型

```bash
cd /mnt/data
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
```
### 推理测试脚本

保存为 `run_qwen_cpu.py`

```python
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

model_name = "/mnt/data/Qwen-7B-Chat"
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto").eval()

inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

执行：

```bash
python run_qwen_cpu.py
```

## 2.2 ChatGLM3-6B 本地部署

### 1. 创建 Conda 虚拟环境

```bash
conda create -n glm_env python=3.10 -y
conda activate glm_env
```

### 2. 安装 PyTorch + torchvision（CPU）

```bash
pip install torch==2.6.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```
### 3. 安装依赖

```bash
pip install transformers==4.33.3
pip install sentencepiece accelerate tqdm
pip install modelscope
```
### 4.下载模型代码

```bash
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
cd chatglm3-6b
```

### 5.推理测试脚本

保存为 `run_chatglm3.py`

```python
from modelscope import AutoTokenizer, AutoModel, snapshot_download

model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision="v1.0.0")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).float().eval()

response, history = model.chat(tokenizer, "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少", history=[])
print("Bot:", response)


执行：

```bash
python run_chatglm3.py
```

---



## 3.1 测试案例设计
设计了以下测试问题来评估模型的语言理解和推理能力：

```python
test_questions = [
    {
        "id": 1,
        "question": "请说出以下两句话区别在哪里？\n1、冬天：能穿多少穿多少\n2、夏天：能穿多少穿多少",
        "category": "语言理解-歧义消解",
        "language": "中文"
    },
    {
        "id": 2,
        "question": "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上。"
        "category": "语言理解-句法分析",
        "language": "中文"
    },
    {
        "id": 3,
        "question": "What's the difference between these two sentences?\n1. He said she said he wouldn't say.\n2. She said he said she wouldn't say.",
        "category": "语言理解-句法分析",
        "language": "英文"
    },
]
```

## 4.1结果分析与对比


## 模型能力对比分析

### 1. 语言理解能力

基于测试结果，我们可以从以下维度对比各模型：

#### 中文理解能力
- **歧义消解**：测试模型对"能穿多少穿多少"在不同季节语境下的理解
- **句法分析**：测试模型对复杂中文句式的解析能力

#### 英文理解能力
- **语法结构**：测试模型对英文复杂句式的理解
- **语义歧义**：测试模型对同形异义句子的区分能力

## 结论与建议

### 主要发现

1. **模型性能差异**：
   - 各模型在不同类型的语言理解任务上表现存在差异
   - 响应时间与模型复杂度相关

2. **应用场景适配**：
   - 中文场景：[根据实际测试结果填写]
   - 英文场景：[根据实际测试结果填写]
   - 推理任务：[根据实际测试结果填写]



## 参考资源

- [魔搭平台官方文档](https://modelscope.cn/docs)
- [通义千问官方文档](https://github.com/QwenLM/Qwen)
- [ChatGLM3官方文档](https://github.com/THUDM/ChatGLM3)
- [百川2官方文档](https://github.com/baichuan-inc/Baichuan2)


**注意**: 本项目仅用于学习和研究目的，请遵守各模型的使用协议和相关法律法规。 # -
