# 魔搭平台大语言模型部署与对比测试项目

## 项目概述

本项目旨在通过魔搭平台（ModelScope）部署和对比测试三个主流大语言模型，包括：
- **通义千问Qwen-7B-Chat**
- **智谱ChatGLM3-6B** 
- **百川2-7B-对话模型**

通过标准化的测试案例，对这些模型的语言理解能力、对话质量和推理性能进行横向对比分析。

## 环境准备

### 1. 注册并登录魔搭平台

1. 访问魔搭平台官网：https://modelscope.cn/
2. 点击右上角"登录"按钮
3. 选择"阿里云账号登录"
   
### 2. 创建Notebook环境

1. 在魔搭平台主页，点击"创建空间"
2.  点击"创建"并等待环境启动

## 模型部署流程

### 1. 环境配置

在Jupyter Notebook中执行以下代码安装必要依赖：

```python
# 安装ModelScope SDK
!pip install modelscope

# 安装其他必要依赖
!pip install torch transformers tokenizers
!pip install pandas numpy matplotlib seaborn
!pip install jupyter notebook
```

### 2. 模型下载与加载

#### 2.1 通义千问Qwen-7B-Chat

```python
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

# 模型本地路径
model_name = "/mnt/data/Qwen-7B-Chat"

# 输入提示
prompt = "What's the difference between these two sentences?\n"
    "1. He said she said he wouldn't say.\n"
    "2. She said he said she wouldn't say."

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 加载模型并设为评估模式
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"  # 自动选择 float32 或 float16，依据模型配置
).eval()

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").input_ids
)
# 推理生成
outputs = model.generate(inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

#### 2.2 智谱ChatGLM3-6B

```python
from modelscope import AutoTokenizer, AutoModel, snapshot_download

# 下载模型（首次使用时执行，模型会缓存至 ~/.cache/modelscope/hub/）
model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision="v1.0.0")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True
)

# 加载模型并转换为 float32（避免在 CPU 上使用 half 精度导致报错）
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True
).float()

# 设置模型为推理模式
model = model.eval()

# 单轮对话测试
response, history = model.chat(
    tokenizer,
    "What's the difference between these two sentences?\n"
    "1. He said she said he wouldn't say.\n"
    "2. She said he said she wouldn't say.",
    history=[]
)

# 输出结果
print(response)
```

#### 2.3 百川2-7B-对话模型

```python
# 下载并加载百川2-7B-Chat模型
model_name_baichuan = "baichuan-inc/Baichuan2-7B-Chat"
tokenizer_baichuan = AutoTokenizer.from_pretrained(model_name_baichuan, trust_remote_code=True)
model_baichuan = AutoModelForCausalLM.from_pretrained(
    model_name_baichuan,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### 3. 模型推理函数

```python
def generate_response(model, tokenizer, prompt, max_length=2048):
    """
    通用的模型推理函数
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()
```

## 测试案例设计

### 测试问题集

我们设计了以下测试问题来评估模型的语言理解和推理能力：

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

### 测试执行代码

```python

   
```

## 结果分析与对比

### 1. 基础统计分析

```python
# 显示基本统计信息
print("=== 测试结果统计 ===")
print(f"总测试数量: {len(results_df)}")
print(f"成功测试数量: {len(results_df[results_df['status'] == 'success'])}")
print(f"失败测试数量: {len(results_df[results_df['status'] == 'error'])}")

# 按模型分组统计
model_stats = results_df.groupby('model').agg({
    'response_time': ['mean', 'median', 'std'],
    'status': lambda x: (x == 'success').sum()
}).round(3)

print("\n=== 各模型性能统计 ===")
print(model_stats)
```

### 2. 响应时间对比

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 响应时间对比图
plt.figure(figsize=(12, 6))
sns.boxplot(data=results_df[results_df['status'] == 'success'], 
            x='model', y='response_time')
plt.title('各模型响应时间对比')
plt.ylabel('响应时间 (秒)')
plt.xlabel('模型')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 3. 详细结果展示

```python
# 按问题分类展示结果
for question_id in sorted(results_df['question_id'].unique()):
    question_data = results_df[results_df['question_id'] == question_id]
    
    print(f"\n{'='*80}")
    print(f"问题 {question_id}: {question_data.iloc[0]['category']}")
    print(f"{'='*80}")
    print(f"问题内容: {question_data.iloc[0]['question']}")
    print(f"\n各模型回答对比:")
    
    for _, row in question_data.iterrows():
        print(f"\n【{row['model']}】")
        print(f"回答: {row['response']}")
        print(f"响应时间: {row['response_time']:.2f}秒")
        print(f"状态: {row['status']}")
        print("-" * 60)
```

## 模型能力对比分析

### 1. 语言理解能力

基于测试结果，我们可以从以下维度对比各模型：

#### 中文理解能力
- **歧义消解**：测试模型对"能穿多少穿多少"在不同季节语境下的理解
- **句法分析**：测试模型对复杂中文句式的解析能力

#### 英文理解能力
- **语法结构**：测试模型对英文复杂句式的理解
- **语义歧义**：测试模型对同形异义句子的区分能力

### 2. 推理逻辑能力

```python
def analyze_reasoning_ability(results_df):
    """
    分析各模型的推理能力
    """
    reasoning_analysis = {}
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        # 计算各类别的成功率
        category_success = model_data.groupby('category').apply(
            lambda x: (x['status'] == 'success').sum() / len(x)
        )
        
        reasoning_analysis[model] = {
            '语言理解-歧义消解': category_success.get('语言理解-歧义消解', 0),
            '语言理解-句法分析': category_success.get('语言理解-句法分析', 0),
            '平均响应时间': model_data['response_time'].mean(),
            '总体成功率': (model_data['status'] == 'success').sum() / len(model_data)
        }
    
    return reasoning_analysis

reasoning_results = analyze_reasoning_ability(results_df)
```

### 3. 综合评估矩阵

```python
def create_evaluation_matrix(reasoning_results):
    """
    创建综合评估矩阵
    """
    eval_df = pd.DataFrame(reasoning_results).T
    
    # 添加综合评分（基于多个指标的加权平均）
    eval_df['综合评分'] = (
        eval_df['语言理解-歧义消解'] * 0.3 +
        eval_df['语言理解-句法分析'] * 0.3 +
        eval_df['总体成功率'] * 0.4
    ) * 100
    
    # 响应时间评分（时间越短分数越高）
    max_time = eval_df['平均响应时间'].max()
    eval_df['响应速度评分'] = (max_time - eval_df['平均响应时间']) / max_time * 100
    
    return eval_df.round(2)

evaluation_matrix = create_evaluation_matrix(reasoning_results)
print("=== 综合评估矩阵 ===")
print(evaluation_matrix)
```

## 结论与建议

### 主要发现

1. **模型性能差异**：
   - 各模型在不同类型的语言理解任务上表现存在差异
   - 响应时间与模型复杂度相关

2. **应用场景适配**：
   - 中文场景：[根据实际测试结果填写]
   - 英文场景：[根据实际测试结果填写]
   - 推理任务：[根据实际测试结果填写]

### 使用建议

1. **选择标准**：
   - 如果注重响应速度：选择响应时间最短的模型
   - 如果注重准确性：选择综合评分最高的模型
   - 如果注重中文理解：选择中文测试表现最佳的模型

2. **部署建议**：
   - 生产环境建议使用成功率最高的模型
   - 开发测试可以使用响应最快的模型
   - 特定场景可以根据专项测试结果选择

## 项目文件结构

```
hw4/
├── README.md                 # 项目说明文档
├── notebooks/
│   ├── model_deployment.ipynb   # 模型部署代码
│   ├── testing_framework.ipynb # 测试框架代码
│   └── results_analysis.ipynb  # 结果分析代码
├── results/
│   ├── test_results.csv         # 测试结果数据
│   ├── evaluation_matrix.csv    # 评估矩阵
│   └── performance_charts/      # 性能对比图表
├── data/
│   └── test_questions.json      # 测试问题集
└── utils/
    ├── model_utils.py          # 模型工具函数
    └── analysis_utils.py       # 分析工具函数
```



## 参考资源

- [魔搭平台官方文档](https://modelscope.cn/docs)
- [通义千问官方文档](https://github.com/QwenLM/Qwen)
- [ChatGLM3官方文档](https://github.com/THUDM/ChatGLM3)
- [百川2官方文档](https://github.com/baichuan-inc/Baichuan2)


**注意**: 本项目仅用于学习和研究目的，请遵守各模型的使用协议和相关法律法规。 # -
