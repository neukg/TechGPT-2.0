# TechGPT 2.0: Technology-Oriented Generative Pretrained Transformer 2.0
Demo: [TechGPT-neukg](http://techgpt.neukg.com) <br>
HuggingFace🤗: [neukg/TechGPT-7B](https://huggingface.co/neukg)

<div align="center">

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/neukg/TechGPT/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/neukg)
</div>

## 引言
随着大模型时代的到来，大模型与知识图谱融合的工作日益成为当前研究的热点。为了对这项工作提供研究的基础，东北大学知识图谱研究组此前发布了[TechGPT-1.0](https://github.com/neukg/TechGPT)大模型。经过了几个月的工作，东北大学知识图谱研究组发布**TechGPT-2.0**大模型，共包含两个7B模型，并同时扩充了多项能力。
## 内容导引
| 章节                   | 描述                          |
|----------------------|-----------------------------|
| [💁🏻‍♂️模型简介](#模型简介) | 简要介绍本项目 TechGPT 2.0 模型的技术特点 |
| [📝模型亮点](#模型亮点)      | 介绍了 TechGPT 2.0 大模型的独特之处    |
| [⏬模型下载](#模型下载)       | TechGPT 2.0 大模型下载地址         |
| [💻环境部署](#推理与部署)     | 介绍了如何使用个人环境部署并体验大模型         |
| [💯系统效果](#系统效果)      | 介绍了模型在部分任务上的效果              |

## 模型简介
TechGPT-2.0 为 TechGPT-1.0 基础上的改进版本，此次共发布两个7B版本的模型分别为**TechGPT2-Alpaca**、**TechGPT2-Atom**。

TechGPT-2.0 较 TechGPT-1.0 新加了许多领域知识。除了 TechGPT-1.0 所具备的计算机科学、材料、机械、冶金、金融和航空航天等十余种垂直专业领域能力，TechGPT-2.0 还在**医学、法律领域**展现出优秀的能力，并扩充了**地理地区、运输、组织、作品、生物、自然科学、天文对象、建筑**等领域能力。除此之外，我们的工作还对**幻觉、不可回答问题、长文本**等任务进行了研究。

**东北大学知识图谱研究组与华为沈阳人工智能计算中心**合作，使用**华为昇腾服务器**（具体为4机*8卡-32G 910A）进行**全量微调**。

## 模型亮点
TechGPT-2.0 在 TechGPT-1.0 的基础上进行了重要的改进，其中最显著的优化是引入了命名实体识别的领域数据，并且使模型具备了**对嵌套实体的抽取能力**。
- 首先，TechGPT-2.0 在医学领域的表达和理解方面取得了显著的提升。模型能够对疾病、药物、专业术语等实体的进行更准确识别，这表明模型在处理医学文本时能够更全面地理解上下文信息。除此之外，TechGPT-2.0 还能够理解医学文本中的复杂关系、疾病诊断、治疗方案等内容。这种全面的医学分析能力使得模型可以用于协助医生阅读医学文献、提供患者诊断建议等应用场景，从而提高医学领域的信息处理效率和准确性。
- 其次，TechGPT-2.0 能够理解和解释法律文本，包括法规、合同和案例法等。这使得模型在法律领域的应用更为广泛，并可以用于解决自动化合同审查、法规遵循检查等任务。模型通过学习法律用语和结构，能够更准确地捕捉文本中的法律关系和条款，为用户提供更有深度和专业性的法律分析。
- 再次，TechGPT 2.0 的另一个重要特性是能够抽取嵌套实体。这意味着模型可以更灵活地处理实体之间的复杂关系，深入挖掘文本中的层次结构，提高了对复杂文本的理解和生成能力。例如，在医学文献中，可能存在嵌套的实体关系，如疾病的亚型、药物的剂量信息等，TechGPT-2.0 能够更好地捕捉这些信息，并在生成回应时更准确地反映上下文的语境。
- 最后，TechGPT-2.0 还具备解决幻觉和人类价值观对齐的能力。模型通过对话和理解上下文，能够更好地理解人类的感受和价值观，并在回应中考虑这些因素。这使得模型能够更好地与人类用户进行交互，更好地满足用户的需求和期望，进一步提升了人机交互的质量和用户体验。
总体而言，TechGPT 2.0 在继承了 TechGPT 1.0 的强大自然语言处理能力的同时，通过增加多领域、多任务的数据，展现出了嵌套实体的抽取、幻觉回答、回答不可回答问题和回答长文本问题的能力。这些改进使得模型更适用于广泛的应用场景，为用户提供了更准确、更深入的信息处理和生成能力。

## 模型下载
| 模型名称                             |  类型  | 训练方式 |  大小   |                              LoRA下载地址                              |
|:---------------------------------|:----:|:----:|:-----:|:------------------------------------------------------------------:|
| TechGPT-1.0                      | 指令模型 | 全量微调 | 13 GB | [[🤗HF]](https://huggingface.co/neukg/TechGPT-7B)  |
| TechGPT-2.0-Alpaca 🆕            | 指令模型 | 全量微调 | 13 GB |  [[🤗HF]](https://huggingface.co/neukg/TechGPT-2.0-alpaca-hf)  |
| TechGPT-2.0-Atom 🆕              | 指令模型 | 全量微调 | 13 GB | [[🤗HF]](https://huggingface.co/neukg/TechGPT-2.0-atom-hf) |


## 环境部署
### 在华为昇腾 910 NPU 服务器上的环境要求
- 硬件：Ascend 910A/910B
- Python：3.9
- MindSpore：2.1.1
- CANN: 6.3.0 RC2
- MindFormers版本：dev
- 7b 推理可在单机单卡上完成部署 

1. 在mindformers环境下执行推理部署时，需要使用ckpt权重；如果没有ckpt权重，则在mindformers目录下需要运行如下转换脚本，将huggingface权重转为ckpt权重，才能使用NPU进行推理：
``` shell
python mindformers/models/llama/convert_weight.py \
--torch_ckpt_dir TORCH_CKPT_DIR \
--mindspore_ckpt_path {path}/MS_CKPT_NAME
```
2. 初次在mindformers环境下执行推理时，会在```mindspore_inference.py```的同级目录下生成```checkpoint_download```文件夹，其中包含了推理所需的```yaml```配置文件和```tokenizer.model```词表等，需要将词表换成该项目huggingface上的对应词表，并将配置文件替换为```mindspore_inference```目录下的```yaml```文件。

### 在 GPU 服务器上的环境要求
请注意TechGPT2-Alpaca和TechGPT2-Atom模型在**训练**和**推理**阶段所使用的prompt格式是不同。
**TechGPT2-Alpaca 使用的prompt格式为：**
``` python
<s> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: [INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。\n<</SYS>>\n\n{instruction} [/INST] ASSISTANT: 
```
**TechGPT2-Atom 使用的prompt格式为：**
``` python
<s>Human: {instruction} \n</s><s>Assistant: 
```

请在使用TechGPT之前保证你已经安装好`transfomrers`和`torch`：

```shell
pip install transformers
pip install torch
```

- 注意，必须保证安装的 `transformers` 的版本中已经有 `LlamaForCausalLM` 。<br>
- Note that you must ensure that the installed version of `transformers` already has `LlamaForCausalLM`.

[TechGPT2-Alpca Example:](https://github.com/neukg/TechGPT/blob/main/inference.py)

``` python
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import torch


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。"""
TEMPLATE = (
    "<s> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: [INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST] ASSISTANT: "
)

def generate_prompt(instruction):
    return TEMPLATE.format_map({'instruction': instruction, 'system_prompt': DEFAULT_SYSTEM_PROMPT})

ckpt_path = './LLM/TechGPT2-Alpaca-hf/'
load_type = torch.float16
device = torch.device(1)
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"
model_config = AutoConfig.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=load_type, config=model_config)
model.to(device)
model.eval()

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    max_new_tokens=128,
    min_new_tokens=10,
    do_sample=True,
)

example = '请把下列标题扩写成摘要, 不少于100字: 基于视觉语言多模态的实体关系联合抽取的研究。'

instruction = generate_prompt(instruction=example)

instruction = tokenizer(instruction, return_tensors="pt")
input_ids = instruction["input_ids"].to(device)
with torch.no_grad():
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        repetition_penalty=1.2,
    )
    output = generation_output.sequences[0]
    output = tokenizer.decode(output, skip_special_tokens=True)
    print(output)

```

[TechGPT2-Atom Example:](https://github.com/neukg/TechGPT/blob/main/inference.py)

``` python
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import torch


TEMPLATE = (
    "<s>Human: "
    "{instruction} \n</s><s>Assistant: "
)

def generate_prompt(instruction):
    return TEMPLATE.format_map({'instruction': instruction})

ckpt_path = './LLM/TechGPT2-Atom-hf/'
load_type = torch.float16
device = torch.device(1)
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
tokenizer.pad_token_id = 2
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"
model_config = AutoConfig.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=load_type, config=model_config)
model.to(device)
model.eval()

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=2,
    max_new_tokens=128,
    min_new_tokens=20,
    do_sample=True,
)

example = '抽取出下面文本的实体和实体类型：《女人树》，国产电视剧，由导演田迪执导，根据作家子页的原著改编，故事从1947年开始，跨越了解放战争和建国初期两大历史时期，展现了战斗在隐形战线上的人民英雄是如何不惧怕任何危险，不计较个人牺牲，甚至不顾人民内部的误解和生死裁决，都不暴露个人真实身份，至死不渝，与敌人周旋到底的英雄故事。'

instruction = generate_prompt(instruction=example)

instruction = tokenizer(instruction, return_tensors="pt")
input_ids = instruction["input_ids"].to(device)
with torch.no_grad():
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        repetition_penalty=1.2,
    )
    output = generation_output.sequences[0]
    output = tokenizer.decode(output, skip_special_tokens=True)
    print(output)

```