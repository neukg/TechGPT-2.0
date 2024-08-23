# TechGPT 2.0: Technology-Oriented Generative Pretrained Transformer 2.0
Demo: [TechGPT-neukg](http://techgpt.neukg.com) <br>
HuggingFace🤗: [neukg/TechGPT-7B](https://huggingface.co/neukg)

<div align="center">

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/neukg/TechGPT/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/neukg)
</div>

## 引言
随着大模型时代的到来，大模型与知识图谱融合的工作日益成为当前研究的热点。为了对这项工作提供研究基础，东北大学知识图谱研究组于2023年6月发布了 [TechGPT-1.0](https://github.com/neukg/TechGPT) 大模型。

在 TechGPT-1.0 基础上，经过几个月的努力，东北大学知识图谱研究组发布功能更全、性能更好的 **TechGPT-2.0** 大模型。TechGPT-2.0 延续了 TechGPT-1.0 以“**知识图谱构建**”与“**智能问答**”为核心的构建目标，在继承 TechGPT-1.0 全部功能的基础上，**对其多项功能进行了增强**，并**扩充了一些新功能**。
## 更新！！！
* 2024/8/22 TechGPT-2.0更新TechGPT-2.0-Qwen1.5-7b 大模型,初步测试较Alpaca与Atom模型表现良好,且ALpaca与Atom模型参数已更新。
* 2024/8/22 TechGPT-2.0新加Alpaca与Atom模型的WebGUI界面，用户可通过调整Web界面中的参数测试模型问答能力。
## 内容导引
| 章节                   | 描述                          |
|----------------------|-----------------------------|
| [💁🏻‍♂️模型简介](#模型简介) | 简要介绍本项目 TechGPT 2.0 模型的技术特点 |
| [📝模型亮点](#模型亮点)      | 介绍了 TechGPT 2.0 大模型的独特之处    |
| [⏬模型下载与体验](#模型下载与体验) | TechGPT 2.0 大模型下载地址与Demo体验  |
| [💻环境部署](#推理与部署)     | 介绍了如何使用个人环境部署并体验大模型         |
| [💯系统效果](#系统效果)      | 展示了模型在部分任务上的效果              |

## 模型简介
TechGPT-2.0 为 TechGPT-1.0 基础上的改进版本，其采用华为**昇腾服务器**（4机*8卡-32G 910A）、**华为MindSpore框架**、采用**全量微调**方式训练完成。此次共发布两个7B版本的 TechGPT-2.0 模型，分别为**TechGPT2-Alpaca**、**TechGPT2-Atom**。

TechGPT-2.0 较 TechGPT-1.0 新加了许多领域知识。除了 TechGPT-1.0 所具备的计算机科学、材料、机械、冶金、金融和航空航天等十余种垂直专业领域能力，TechGPT-2.0 还在**医学、法律领域**展现出优秀的能力，并扩充了**地理地区、运输、组织、作品、生物、自然科学、天文对象、建筑**等领域文本的处理能力。TechGPT-2.0 还对**幻觉、不可回答、长文本处理**等问题进行了能力增强。

同时，TechGPT-2.0 对部署的硬件要求更低，使用 NVIDIA 4090 单机单卡、或昇腾910A单机单卡就可完成 TechGPT-2.0 模型部署。

## 模型亮点
TechGPT-2.0 在继承了 TechGPT-1.0 的能力上进行了重要的改进，具有以下几项新增亮点功能：
- 首先，TechGPT-2.0 在**医学领域**文本的处理能力取得了显著提升。TechGPT-2.0 在处理医学文本时能够更全面地理解上下文信息，能够对疾病、药物、专业术语等实体的进行更准确识别，还能够理解医学文本中的复杂关系、疾病诊断、治疗方案等内容。这种全面的医学分析能力使得模型可以用于协助医生阅读医学文献、提供患者诊断建议等应用场景，从而提高医学领域的信息处理效率和准确性。
- 其次，TechGPT-2.0 能够理解和解释法律文本，包括法规、合同和案例法等。TechGPT-2.0 通过学习法律用语和结构，能够更准确地捕捉文本中的法律关系和条款，为用户提供更有深度和专业性的法律分析。这使得 TechGPT-2.0 模型在**法律领域**中的应用前景更为广泛，可以用于解决自动化合同审查、法规遵循检查等任务。
- 再次，TechGPT-2.0 增强了对**嵌套实体**。的抽取能力。即 TechGPT-2.0 可以更灵活地处理实体之间的复杂关系，深入挖掘文本中的层次结构，提高了对复杂文本的理解和生成能力。例如，在医学文献中，可能存在嵌套的实体关系，如疾病的亚型、药物的剂量信息等，TechGPT-2.0 能够更好地捕捉这些信息，并在生成回应时更准确地反映上下文的语境。
- 另外，TechGPT-2.0 在**幻觉、不可回答问题、长文本处理、以及与人类价值观对齐方面进行了强化**。TechGPT-2.0通过对话和理解上下文，能够更好地理解人类的感受和价值观，并在回应中考虑这些因素。此外，TechGPT-2.0 能够支持最大长度12K的文本输入，这使得模型能够更好地与人类用户进行交互，更好地满足用户的需求和期望，进一步提升了人机交互的质量和用户体验。
- 最后，TechGPT-2.0 使用**华为昇腾910A服务器**、**华为Mindspore框架**以及Mindformer套件进行开发，采用分布式训练方案，使用服务器数量为4机*8卡（32G），具有更低的部署硬件要求。目前 TechGPT-2.0 的部署在使用**NVIDIA**的显卡并使用float16的情况下显存仅需**15G**，INT8量化下显存占用**8G**，INT4量化下显存仅需**5G**；NVIDIA 4090单机单卡、或昇腾910A单机单卡就可以部署 TechGPT-2.0 模型。

总体而言，TechGPT-2.0 在继承了 TechGPT-1.0 的全部特性的同时，**通过增加多领域、多任务的数据，展现出了嵌套实体的抽取、幻觉回答、回答不可回答问题和回答长文本问题的能力**。这些改进使得模型更适用于广泛的应用场景，为用户提供了更准确、更深入的信息处理和生成能力。

## 模型下载与体验
### 下载地址
| 模型名称                     |  训练方式  |  大小   |                        HuggingFace 下载                        |                               wisemodel 下载                                |                                                                           ModelScope 下载                                                                           |
|:-------------------------|:------:|:-----:|:------------------------------------------------------------:|:-------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| TechGPT-1.0              |  全量微调  | 13 GB |      [[🤗HF]](https://huggingface.co/neukg/TechGPT-7B)       |  [[wisemodel社区]](https://www.wisemodel.cn/models/undefined/TechGPT-1.0)   |    [[<img src="https://g.alicdn.com/sail-web/maas/1.10.1/static/modelscopeIcon.cd89353f.svg" height="12">]](https://modelscope.cn/models/neukg01/TechGPT-1.0)     |
| TechGPT-2.0-Alpaca 🆕    |  全量微调  | 13 GB | [[🤗HF]](https://huggingface.co/neukg/TechGPT-2.0-alpaca-hf) | [[wisemodel社区]](https://www.wisemodel.cn/models/neukg/TechGPT-2.0-Alpaca) | [[<img src="https://g.alicdn.com/sail-web/maas/1.10.1/static/modelscopeIcon.cd89353f.svg" height="12">]](https://modelscope.cn/models/neukg01/TechGPT-2.0-Alpaca) |
| TechGPT-2.0-Atom 🆕      |  全量微调  | 13 GB |  [[🤗HF]](https://huggingface.co/neukg/TechGPT-2.0-atom-hf)  |  [[wisemodel社区]](https://www.wisemodel.cn/models/neukg/TechGPT-2.0-Atom)  |  [[<img src="https://g.alicdn.com/sail-web/maas/1.10.1/static/modelscopeIcon.cd89353f.svg" height="12">]](https://modelscope.cn/models/neukg01/TechGPT-2.0-Atom)  |
| TechGPT-2.0-QLora 🆕     | Lora微调 | 3 GB  | [[🤗HF]](https://huggingface.co/neukg/TechGPT-2.0-QLora-hf)  | [[wisemodel社区]](https://www.wisemodel.cn/models/neukg/TechGPT-2.0-QLora)  | [[<img src="https://g.alicdn.com/sail-web/maas/1.10.1/static/modelscopeIcon.cd89353f.svg" height="12">]](https://modelscope.cn/models/neukg01/TechGPT-2.0-QLora)  |
| TechGPT-2.0-Qwen1.5 🆕   |  全量微调  | 14 GB | [[🤗HF]](https://huggingface.co/neukg/TechGPT-2.0-Qwen1.5-7b)  | [[wisemodel社区]](https://www.wisemodel.cn/models/neukg/TechGPT-2.0-Qwen1.5)  | [[<img src="https://g.alicdn.com/sail-web/maas/1.10.1/static/modelscopeIcon.cd89353f.svg" height="12">]](https://modelscope.cn/models/neukg01/TechGPT-2.0-Qwen1.5-7b)  |

### 模型说明
**TechGPT2** 在不同规模且经过扩充后的**TechKG**大规模的中文学术语料支持下训练完成。
- **TechGPT2-Alpaca** 使用HFL发布的Chinese-Alpaca-2-7B模型，Chinese-Alpaca-2-7B模型是在Chinese-LLaMA-2-7B的基础上进一步通过指令精调获得。Chinese-LLaMA-2-7B为基座模型，是在LLaMA-2的基础上进行中文增量训练获得。
- **TechGPT2-Atom** 使用Llama中文社区联合AtomEcho（原子回声）共同发布的Atom-7B-chat，并在此基础上进行全量微调后获得。
- 此外，我们使用**QLora线性插值法**微调的长文本模型的Lora权重也已经发布在huggingface上，请自行下载合并。

我们TechGPT-2的两个**7B**版本的模型、以及使用**QLora线性插值法**微调的长文本模型都已经在Hugging Face和GitHub上开源，后续在这些模型基础上的改进，也将开源到相同账号，欢迎大家使用并提出宝贵的意见。

### 模型体验
我们目前对外提供 TechGPT-1.0 与 TechGPT2-Atom 版本的在线服务：http://techgpt.neukg.com/

作为一个学术组织，我们无法长期提供模型的在线服务功能，当前的体验系统存在着随时下线的可能。因此建议大家后续通过开源权重自行体验，共同创建更好的中文大模型开源环境。

## 环境部署
### 在华为昇腾 910 NPU 服务器上的环境要求
- 硬件：Ascend 910A/910B
- Python：3.9
- MindSpore：2.1.1
- CANN: 6.3.0 RC2
- MindFormers版本：dev
- 7b 推理可在单机单卡上完成部署 

1. 在mindformers环境下执行推理部署时，需要使用ckpt权重；如果没有ckpt权重，则在mindformers目录下需要运行如下[转换脚本](https://github.com/neukg/TechGPT-2.0/blob/main/ckpt_weight_convert/convert_weight.py)，将huggingface权重转为ckpt权重，才能使用NPU进行推理：
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

[TechGPT2-Alpca Example:](https://github.com/neukg/TechGPT-2.0/blob/main/pytorch_inference/techgpt2-alpaca_infer.py)

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

[TechGPT2-Atom Example:](https://github.com/neukg/TechGPT-2.0/blob/main/pytorch_inference/techgpt2-atom_infer.py)

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

## 系统效果
TechGPT-2.0 在保留了 TechGPT-1.0 模型的通用能力、实体识别能力以及三元组抽取能力的基础上，能够完成**更为复杂的嵌套实体识别**，并在**医学、法律领域**展现出优秀的能力，此外对**幻觉、不可回答问题以及长文本问题**有着较好的解决能力。

下面展示了TechGPT-2.0 的几个新增能力的样例。

1. **医学NER**
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>这句话中是否有实体？如果有，请列举出来。<br/>依据主要病变部位分为下列数型：脊髓型最常见，为脊髓前角细胞受损所致，表现为分布不对称、不规则的弛缓性肌肉瘫痪，四肢多见，下肢尤甚。 </td>     <td>实体名:先天畸形, 实体类型:疾病<br/>实体名:发育障碍, 实体类型:临床表现<br/>实体名:上、下呼吸道梗阻, 实体类型:疾病<br/>实体名:膈疝, 实体类型:疾病<br/>实体名:肺部受压迫, 实体类型:临床表现<br/>实体名:肺, 实体类型:身体<br/>实体名:呼吸衰竭, 实体类型:疾病
</td></tr>
<tr><td>毒蕈类以1∶5000高锰酸钾或活性炭悬液反复洗胃，硫酸镁导泻；用二巯基丙磺酸钠每次5mg/kg，每天2～3次，肌内注射；阿托品0.05mg/kg，肌内注射，每15分钟1次，直至阿托品化后减量。<br/>在这句话中，是否可以找到任何实体名称？ </td>     <td>实体名:毒蕈, 实体类型:微生物类<br/>实体名:高锰酸钾, 实体类型:药物<br/>实体名:活性炭, 实体类型:药物<br/>实体名:硫酸镁, 实体类型:药物<br/>实体名:导泻, 实体类型:医疗程序<br/>实体名:二巯基丙磺酸钠, 实体类型:药物<br/>实体名:肌内注射, 实体类型:医疗程序<br/>实体名:阿托品, 实体类型:药物<br/>实体名:肌内注射, 实体类型:医疗程序<br/>实体名:阿托品, 实体类型:药物
</td></tr>
</table>

2. **嵌套NER**
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>一位原招商银行信用卡中心负责人告诉记者。请抽取出其中的命名实体。 </td>     <td>实体名: 招商银行，实体类型：企业<br/>实体名: 中国，实体类型：地点<br/>实体名: 招商银行信用卡中心， 实体类型：机构<br/>实体名: 银行，实体类型：企业
</td></tr>
<tr><td>这句话中是否有实体？如果有，请列举出来。<br/>依据主要病变部位分为下列数型：脊髓型最常见，为脊髓前角细胞受损所致，表现为分布不对称、不规则的弛缓性肌肉瘫痪，四肢多见，下肢尤甚。</td>     <td>实体名:脊髓, 实体类型:身体<br/>实体名:脊髓前角细胞, 实体类型:身体<br/>实体名:脊髓前角细胞受损, 实体类型:临床表现<br/>实体名:分布不对称、不规则的弛缓性肌肉瘫痪, 实体类型:疾病<br/>实体名:肢体, 实体类型:身体<br/>实体名:下肢, 实体类型:身体
</td></tr>
<tr><td>这句话是否包含任何命名实体？如果有，请将其列举出来。<br/>在弥漫性肺损伤和肺泡萎陷时，要获得比较好的氧合，需要在机械通气时保持压力的平台，以使肺泡持续扩张，气体在不同时间常数的肺泡单位内移动，达到压力平台，使得到较好的肺泡通气/灌流，减少肺内分流。</td>     <td>实体名:弥漫性肺损伤, 实体类型:疾病<br/>实体名:肺泡萎陷, 实体类型:临床表现<br/>实体名:肺泡, 实体类型:身体<br/>实体名:机械通气, 实体类型:医疗程序<br/>实体名:压力的平台, 实体类型:医疗设备<br/>实体名:肺泡持续扩张, 实体类型:临床表现<br/>实体名:气体不同时间常数, 实体类型:医学检验项目<br/>实体名:肺泡通气/灌流, 实体类型:医疗程序<br/>实体名:肺内分流, 实体类型:临床表现<br/>实体名:肺内, 实体类型:身体
</td></tr>
<tr><td>出血因凝血因子大量被消耗、血小板减少及继发纤溶亢进，发生出血。<br/>请列示这段文本中的所有特定实体。</td>     <td>实体名:出血, 实体类型:临床表现<br/>实体名:凝血因子, 实体类型:身体<br/>实体名:凝血因子大量被消耗, 实体类型:临床表现<br/>实体名:血小板, 实体类型:身体<br/>实体名:血小板减少, 实体类型:临床表现<br/>实体名:继发纤溶亢进, 实体类型:疾病<br/>实体名:出血, 实体类型:临床表现
</td></tr>
</table>

3. **泛化NER（根据背景知识泛化的NER）**
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>下面句子中的实体有哪些《女人树》，国产电视剧，由导演田迪执导，根据作家子页的原著改编，故事从1947年开始，跨越了解放战争和建国初期两大历史时期，展现了战斗在隐形战线上的人民英雄是如何不惧怕任何危险，不计较个人牺牲，甚至不顾人民内部的误解和生死裁决，都不暴露个人真实身份，至死不渝，与敌人周旋到底的英雄故事。 </td>     <td>实体名: 女人树，实体类型：影视作品<br/>实体名: 田迪，实体类型：人物<br/>实体名: 中国，实体类型：国家<br/>实体名: 女人树，实体类型：图书作品<br/>实体名: 子页，实体类型：人物
</td></tr>
<tr><td>俄德两国4架战机全部逼近芬兰空中边界，芬兰空军歼击机迅速起飞干预，驱逐俄德两国歼击机，并将其“护送”。请抽取出其中的命名实体。 </td>     <td>实体名: 俄罗斯，实体类型：国家<br/>实体名: 德国，实体类型：国家<br/>实体名: 芬兰，实体类型：地点<br/>实体名: 芬兰空军，实体类型：机构/人物<br/>实体名: 芬兰，实体类型：国家
</td></tr>
</table>

4. **法律案件梳理**
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>苏国平签订的劳动合同期限是什么？<br/>经审理查明,原告苏国平于1979年4月顶替父亲进入常州东方印染厂(国有企业)工作,1996年6月与常州东方印染厂与华源集团合资成立的常州华昌染工有限公司签订劳动合同,合同期限为1996年6月1日至1997年5月31日。后双方续签,最后一份劳动合同期限为1998年5月31日至2001年5月31日。1999年8月30日,常州华昌染工有限公司作出华昌(1999)3号《关于苏国平开除出公司的决定》,该决定载明“苏国平原系华昌染工有限公司前道主管……是公司多年培养的技术骨干,于99年8月初为了个人利益置公司利益而不顾,未办任何手续不告而别,直接为其他印染厂提供技术服务,使本公司利益受到严重伤害……经公司经理会研究决定将苏国平开除出公司,并要求其负经济赔偿责任”,该决定还载明“抄送:市仲裁处、市纺工局、市郊区劳服公司、本人”。1999年9月,常州华昌染工有限公司由常州东方印染厂托管,东方印染厂劳资经办人员在1999年11月办理解除劳动合同手续时,在解除原因一栏填写了“除名”。苏国平的社保参保缴费证明显示其1999年8月前在常州华昌染工有限公司参保、1999年10月起改至常州市金田纺织印染厂,至2019年2月累计缴费31年1月。2019年2月20日,常州视音电子科技有限公司为原告申办企业职工基本养老保险待遇申领手续,被告于2月25日作出本案退休待遇审核表。原告不服,诉至本院。庭审中,各方当事人围绕案涉开除决定是否合法有效各执己见。</td>     <td>1996年6月1日至1997年5月31日;1998年5月31日至2001年5月31日
</td></tr>
<tr><td>承包方何时进行行贿？<br/>经审理查明:2012年7月,被告人陈x1以不具备承建资质的衡南县向阳镇水口基建队的名义承包了国有事业单位衡南县教师进修学校学员宿舍改造土建项目工程。为获取不正当利益,被告人陈x1于2012年9月底至2013年底分三次送给该校校长、党支部书记程x9好处费共计6万元。具体犯罪事实如下:1、2012年9月份的一天,被告人陈x1为感谢程x9将衡南县进修学校学员宿舍楼改造改建工程交给其做,争取程x9对其做工程给予关照,事后多安排工程给其做,在程x9的办公室送给程x91万元现金,程x9予以收受。2、2012年底的一天,被告人陈x1为承建衡南县进修学校罕舍楼改造的其他后续工程,在衡南县教师进修学校宿舍改造工地上送给程x91万元现金,程x9予以收受,并将后续工程全部交由陈x1承建。3、2013年底的一天,被告人陈x1为感谢程x9在其承建该校工程中给予的关照并争取被告人程x9同意及时尽多拨付工程款,在程x9的办公室送给程4万元现金,程x9予以收受,并尽可能地拨付工程款给陈x1。2014年3月,程x9得知衡南县纪律检查委员会在对其违纪违法行为进行调查后,于3月18日安排其妻子秦某某将从被告人陈x1处收受的6万元贿赂款退还给了陈x1。2014年5月12日,被告人陈x1主动向湖xxxxxxxxx0投案,如实供述了自己的犯罪事实,并上缴了其用于行贿的6万元钱。上述事实有被告人陈x1于衡南县教师进修学校签订的建筑安装工程施工合同、被告人陈x1的常口信息等书证;同案犯程x9的供述及证人秦某某、罗某某等人的证言;被告人陈x1的供述与辩解等证据证实,经庭审质证,被告人陈x1均无异议,足以认定。 </td>     <td>2012年9月份<br/>2012年底<br/>2013年底
</td></tr>
<tr><td>案中法院作出过什么民事判决书<br/>经审理查明,原、被告曾7合伙关系,双方因纠纷经本院调解达成(2014)泰兴戴民初字第0746号民事调解书,被告龚x1依调解协议约定,于2015年2月12日将应支付给原告的19万元缴纳至本院。2015年2月12日,被告龚x1向本院提起诉讼,要求案外人窦x6立即给付其货款16万元,并将本案原告列为该案第三人。2015年3月27日,依龚x1的申请,本院作出(2015)泰兴商初字第123号民事裁定书,对原告的19万元(被告缴纳至本院的)进行了保全。2015年4月23日,本院作出(2015)泰兴商初字第123号民事判决书,判决:1、窦x6支付龚x1、兴化市星权化工产品经营部31003.4元;2、驳回龚x1、兴化市星权化工产品经营部的其他诉讼请求。后因龚x1不服,上诉至泰州市中级人民法院,该院于2016年1月5日作出(2015)泰中商四终字第73号民事判决:驳回上诉,维持原判。2016年2月5日,本院解除了对原告财产的保全,原告亦于当日取得了19万元。另查明,涉案的19万元系被告应支付给原告的履行款,由被告缴纳至兴化市财政局在江苏吴江农村商业银行股份有限公司开设的账户内。2015年3月27日,中国人民银行同期贷款利率为年利率5.35%,江苏吴江农村商业银行同期活期存款利率为年利率0.35%。以上事实有原告提供的民事调解书、民事判决书、民事裁定书、被告提供的现金解款单及本院庭审笔录在卷佐证。 </td>     <td>（2015）泰兴商初字第123号民事判决书<br/>（2015）泰中商四终字第73号民事判决
</td></tr>
</table>

5. **法律咨询**
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>夫妻双方已经离婚，有一个儿子，因为丈夫犯了错，老人明确写下了与丈夫断绝关系，财产由另外两个子女平分，多年来妻子从未过问大家庭一次，老人生病也连一句问候都没有，现在老人走了，妻子有权利争夺财产吗？</td>     <td>没有权利。根据你的描述，老人的遗嘱是明确的将遗产留给了其他兄弟姐妹，而你并没有继承权；同时你作为丈夫的前妻和儿子的母亲也没有任何的法定继承人资格来分割遗产。因此，你没有权利去争抢这个遗产份额。
</td> </tr>
<tr><td>你好，请问在工地干活左手手腕筋断一根和断血管一根能赔多少钱？ </td>     <td>工伤赔偿项目包括：医疗费、住院伙食补助费、停工留薪期工资、护理费、交通费等。具体赔偿金额需要根据实际情况来确定的。建议您先申请劳动能力鉴定，确定伤残等级后按照伤残等级计算赔偿数额。
</td> </tr>
<tr><td>无证醉酒驾驶摩托车，喝了一杯白酒，怎么处罚？ </td>     <td>饮酒后驾车属于危险驾驶行为。根据《中华人民共和国道路交通安全法》的规定：未取得机动车驾驶证、机动车驾驶证被吊销或者机动车驾驶证被暂扣期间驾驶机动车的处二百元以上二千元以下罚款，可以并处十五日以下拘留。<br/>同时，如果造成事故构成犯罪的，还要承担刑事责任。具体来说，酒后驾驶情节严重的，可能会面临拘役的处罚；而醉驾致人死亡或重伤的，则可能会被判处三年以上七年以下的有期徒刑。因此，建议您不要酒驾，以免给自己和他人带来不必要的伤害和损失。
</td> </tr>
</table>

6. **医学咨询**
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>年10岁近两天再次出现干呕恶心呕吐腹痛，喝了止泻药但情况没太大变动，应该如何治疗，如何吃晚饭？</td>     <td>您描述的症状主要包括胃部不适症状如恶心、呕吐以及肚子痛，建议立即就医确诊病因并采取合适的治疗方法。以下是一些可能帮助您的方法和建议：<br/>* 首先应该考虑是否是因为最近吃了什么不健康的食物或饮料导致消化不良所引起。如果是这样，可以避免这些食物以促进消化吸收。此外，可以考虑喝一杯温水以刺激肠胃运动，或者使用小苏打来缓解酸碱失衡问题（请咨询医生）。<br/>* 如果以上方案无效或是其他原因引起的疼痛等病症持续存在并且影响日常生活时间较长的话最好去医院检查一下是否存在某种疾病感染，进行相应的处理方式会因具体病情而异。同时需要评估治疗费用负担的能力状况，以免造成经济损失。<br/>* 在生病期间饮食选择上需要注意营养均衡的选择多样化多样的食品组合，避免过量或不当搭配的情况产生不良反应。例如食用易腐烂变质食物要加强注意；辛辣刺激性食物需减少甚至禁食等等。
</td> </tr>
<tr><td>邻居家的女宝宝才1岁，近几天，孩子的妈妈说孩子耳朵有点疼，察觉到，偶尔还会有黄色的液体流出，同时，食欲也不好，嗓子也不舒服，请问：小孩儿中耳炎耳朵胀痛怎样治效果好。 </td>     <td>抗生素药物是目前治疗中耳炎比较常用的，可酌情选。也可以配合一些局部治疗的方法，比如消炎型的滴耳液医治，如孩子耳痛流黄水的症状严重，建议选用一些镇痛药缓解症状，要是伴随发高烧的情况，那么根据孩子的症状使用药物，严重的情况请尽快去医院进行救治，存在耳膜穿孔的情况家长可考虑可孩子手术治疗，期间注意让孩子充分休息，并且多吃许多蔬菜水果。
</td> </tr>
<tr><td>早期胃癌手术四十天了，其它都正常，就是夜里有点虚汗，平时吃稀饭和面条以及鱼汤或骨头汤等，现在可以吃点米饭吗？饮食上吃点什么可以增强营养。 </td>     <td>病情分析：您好！根据您的描述来看，您目前情况考虑为术后恢复期的问题指导意见：建议您可以适当进食一些易消化的食物，如粥、面片、馄饨等，同时注意补充蛋白质及维生素C的摄取，以促进伤口愈合，提高免疫力，预防感染的发生。祝您早日康复！
</td> </tr>
</table>

7. **幻觉问题**
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>地球上一共装了多少台行星发动机？</td>     <td>实际上，地球并没有安装任何的“行星引擎”。这是一个常见的误解或谣言来源之一。以下是一些详细的解释：<br/>首先需要明确的是，我们通常所说的“行星引擎”指的是一种人造卫星上的推进系统，它通过使用燃料和氧化剂来产生推力以使卫星在轨道上运动并维持其运行状态的一种装置。这些推进系统的类型包括化学火箭、核反应堆等。然而，这种类型的设备并不存在于地球的表面或者大气层中。相反地，它们是由人类设计和建造的用于太空探索和研究的工具。因此，我们不能说地球上安装了任何形式的“行星引擎”。<br/>其次，即使将这个问题看作是指地球上存在某种类似于外星文明的先进技术，我们也无法确定该文明是否真的拥有类似星际飞船这样的交通工具以及它们的运作方式是什么样的。这涉及到许多科学和技术难题，如宇宙物理学、生命起源等等，而这些问题的答案目前还没有被完全解决。 因此，关于外星人存在的证据仍然非常有限且难以证明，而所谓的“行星引擎”也只是虚构的概念而已。<br/>综上所述，没有确切的数据表明地球上有安装在地面上的任何“行星引擎”的存在。这只是一个流传已久但无根据的说法。
</td> </tr>
<tr><td>中国最好的烹饪学校是哪里？ </td>     <td>没有一个“最好”的烹饪学校，因为每个人对“好”的定义不同。但是以下是一些受欢迎和有影响力的学校：<br/>1. 北京厨艺学院（Beijing Culinary College）- 该校是中国最古老的厨师培训机构之一，成立于1956年。它提供各种课程，包括中餐、西餐、糕点等。该学校的师资力量强大，许多学生毕业后成为知名餐厅的主厨或餐饮企业高管。<br/>2. 上海交通大学附属高级职业技术学院（Shanghai Jiao Tong University Affiliated Vocational Technical School of Advanced Professional Technology and Arts） - 这所院校位于上海，是一所综合性的职业教育机构。该校拥有多个专业领域，其中包括烹饪艺术。该校注重实践教学，为学生提供了丰富的实习机会和实践经验。<br/>3. 广东省粤菜文化研究院（Guangdong Province Guangzhou Chinese Cookery Culture Research Institute）- 这是一所专注于广东菜肴文化的研究机构和培训中心。该校的课程涵盖了从基础到高端的菜肴制作技巧，以及中国传统饮食文化和历史背景等方面的知识。该校还与多家著名酒店合作，为学员提供实际操作的机会。<br/>4. 四川成都川味厨房职业技能培训学校（Chengdu Sichuan Kitchen Skills Training School）- 位于四川成都的一家私立学校，专门教授川菜烹饪技艺。该校的教学方法独特，强调理论与实践相结合，让学生能够快速掌握川菜的制作方法和技巧。此外，该校也开设了其他相关领域的课程，如中式面点、点心等。<br/>以上是一些备受欢迎和有影响的烹饪学校，但选择哪个学校取决于个人兴趣和专业需求。在选择时需要考虑学费、地理位置、师资力量等因素。
</td> </tr>
<tr><td>目前公认的最好的职业是什么？ </td>     <td>没有一种职业是“最好”的，因为每个人都有自己的兴趣、技能和目标。然而，以下是一些被认为是最有前途的职业：
<br/>1. 数据科学家/数据分析师 - 随着数字化时代的到来，对数据的需求越来越高，因此数据科学成为了一个非常热门的领域。数据科学家可以帮助企业分析大量数据并做出明智的决策。
<br/>2. 网络安全专家 - 在数字时代，网络攻击变得越来越多样化且复杂。网络安全专业人员需要保护企业和个人免受黑客和其他威胁的攻击。
<br/>3. 医生 - 尽管医疗保健行业面临着许多挑战，但医生仍然是最受欢迎的职业之一。随着人口老龄化和健康意识的提高，人们对医生的需求和期望也在不断增加。
<br/>4. 金融顾问 - 由于金融市场的复杂性和高风险性，金融顾问的需求也日益增长。他们可以为客户提供投资建议和指导，以帮助他们实现财务目标和规划未来。
<br/>5. 开发工程师 - 编程是一项非常有前景的工作，尤其是在技术领域。开发工程师可以在各种行业中工作，包括软件公司、游戏工作室等。
<br/>6. 教师 - 虽然教育行业的竞争激烈，但是教师是一个稳定的职业选择，而且对于那些喜欢与孩子互动的人来说也是一个很好的选择。
<br/>7. 律师 - 作为法律领域的重要角色，律师能够为客户解决问题并提供咨询服务。虽然这个职位的竞争也很大，但它仍然被视为一项很有前途的职业。
<br/>总之，以上这些职业都是目前被广泛认为是最有前途的职业之一。当然，还有很多其他的职业也非常有价值，具体取决于个人的兴趣和能力。
</td> </tr>
</table>

8. **三元组抽取**
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>请给出文中的三元组  《善德女王》是由金根洪、朴弘均联合执导，金英贤编剧，高贤贞、李瑶媛、金南佶、严泰雄、朴艺珍、俞承豪主演的一部韩国历史剧。</td>     <td><善德女王, 主演, 金南佶>
<br/><善德女王, 导演, 金根洪>
<br/><善德女王, 编剧, 金英贤>
<br/><善德女王, 主演, 李瑶媛>
<br/><善德女王, 主演, 高贤贞>
<br/><善德女王, 主演, 严泰雄>
<br/><善德女王, 主演, 朴艺珍>
<br/><善德女王, 导演, 朴弘均>
<br/><善德女王, 主演, 俞承豪>
</td> </tr>
<tr><td>张开建，男，汉族，山东梁山人，1953年11月出生，1969年12月参加工作，1973年9月加入中国共产党，1979年1月北京大学地质系地震地质专业毕业，大学学历，高级工程师<br/>你能否将文本中所有的实体关系三元组列举出来吗？三元组形式是什么呢？ </td>     <td>
<br/><张开建, 民族, 汉族>
<br/><张建, 国籍, 中国>
<br/><张开建, 出生地, 山东梁山>
<br/><张开建, 毕业院校, 北京大学>
<br/><张开建, 出生日期, 1953年11月>
</td> </tr>
<tr><td>请给出下文中的所有关系三元组，三元组形式为<主语，关系，宾语><br/>文本：今天，才知道了国歌的名字叫《义勇军进行曲》，知道了国歌的词曲作者田汉和聂耳，还知道了国歌象征着中华民族永不屈服的抗争精神</td>     <td>
<br/><义勇军进行曲, 作词, 田汉>
<br/><义勇军进行曲, 作曲, 聂耳>
</td> </tr>
</table>

9. **不可回答问题**
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>我想知道2045年美国总统是谁？</td>     <td>无法确定，因为到那时还没有选举。
</td> </tr>
<tr><td>哪个城市的房子一定会升值？ </td>     <td>无法确定，因为房价受多种因素影响。
</td> </tr>
<tr><td>哪只球队在每年的中超联赛中都进球最多？</td>     <td>无法回答，因为没有提供具体的年份信息。需要知道哪个赛季的中超联赛才能确定哪一支球队进球最多。
</td> </tr>
</table>

10. **通用能力**
- 生成标题
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>请用一个简单而有力的标题来概括这篇论文的主要观点，使人们更容易理解它的主旨。
<br/>摘要：单纯降低凿岩爆破的直接成本,而忽视爆破效果不良所造成的掘进效率低、矿石大块率、贫化与损失率提高等,将给生产带来较大的经济损失。金岭铁矿加强爆破工程管理、提高职工技术水平和思想素质,保证爆破效果,提高了矿山的经济效益。
<br/>领域：冶金工业及金属 </td>     <td>浅谈金岭铁矿的爆破效果分析及其对经济效益的影响——以金岭铁矿为例
</td> </tr>
</table>

- 生成摘要
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>标题：基于混合稀疏基字典学习的微波辐射图像重构方法
<br/>领域：信息与通信工程 
<td>为提高雷达目标的检测性能，提出了一种新的基于混合稀疏基(MSG)字典学习算法的微波成像重建模型。该模型利用了信号和噪声之间的相关性来提取出一组特征向量，然后通过在训练样本上建立一个MSG字典对特征向量进行分类，从而实现对目标回波信息的恢复。实验结果表明，所提算法具有较高的鲁棒性、较强的抗噪能力以及较低的计算复杂度，能够有效地改善目标检测效果。
</td> </tr>
</table>

- 其他
<table>   <colgroup>     <col style="width:50%">     <col style="width:50%">   </colgroup>   <tr>     <td>问题</td>     <td>回复</td> </tr>
<tr><td>给一个房产广告写一篇文案，突出该房产的特点和优势。 
</td>     
<td>如果您正在寻找一处舒适、安全且具有投资价值的住宅，那么我们诚挚地邀请您来参观我们的楼盘——XX小区！
<br/>首先，让我们来看看这个区域的地理位置。XX小区位于市中心的黄金地段，交通便利，周边商业配套齐全，生活便捷。同时，这里也是教育资源丰富的区域之一，拥有多所优质学校，为孩子的成长提供了良好的保障。
<br/>其次，我们来看看这个小区的建筑风格。XX小区采用现代化的设计理念，建筑外观简洁大方，内部装修豪华精致，让您感受到家的温馨与舒适。此外，小区内还配备了完善的安保系统，24小时保安巡逻，为您和家人的安全保驾护航。
<br/>最后，让我们来看一下这个小区的投资价值。由于地处繁华商圈，加上优质的物业管理和服务，XX小区的房价一直处于高位，升值潜力巨大。而且，随着城市的发展，未来房价还将持续上涨，是您投资的不二之选。
<br/>总之，XX小区是一个集居住、工作和教育于一体的完美选择。无论是自住还是出租，都是您的理想之地。欢迎前来看房，我们将竭诚为您提供最优质的服务！
</td> </tr>

<tr><td>生成一首诗，以“冬雪”为主题。
</td>     
<td>冬日白雪皑皑落，
<br/>寒风凛冽吹不停。
<br/>银装素裹山岗色，
<br/>美景如画在眼前。
</td> </tr>

<tr><td>发表一篇关于母爱的文章，探讨母爱的力量和影响，至少500字。</td>     
<td>母爱是世间最伟大的力量之一，它像一股温暖的阳光照耀着我们的心灵深处，让我们感受到生命的美好与温暖。母爱是一种无私的奉献、一种无微不至的关怀，它是人类生命中最为珍贵的财富。
<br/>母亲是我们人生中最重要的人之一，她给予我们生命，抚育我们的成长，陪伴我们在人生的道路上前行。母亲的爱如同春雨般滋润着每一个人的心灵，给我们带来无尽的温馨和感动。无论是在生活中还是工作中，我们都需要母亲的支持和理解，因为她们总是默默地为我们付出一切。
<br/>母爱的影响不仅仅体现在情感上，它还能够改变一个人的命运和生活方式。一个有良好家庭教育的孩子往往会更加自信、勇敢、独立，因为他们在成长的过程中得到了父母的关爱和支持；而缺乏母爱的孩子则往往容易变得自卑、内向、孤僻，甚至产生心理问题。因此，母爱对于孩子的成长和发展具有重要的作用。
<br/>同时，母爱也是推动社会进步的重要因素。在一个家庭中，如果父母都能够以身作则地传递爱心和责任感，那么整个社会的道德水平也会得到提高。这种爱的传承不仅可以帮助孩子们树立正确的人生观和价值观，还可以促进家庭的和谐稳定和社会的稳定发展。
<br/>总之，母爱是人类最美好的礼物之一，它不仅仅是一种情感的表达，更是一种责任的体现。作为子女，我们要学会感恩母亲的恩情，珍惜这份宝贵的亲情；作为家长，我们也要时刻关注子女的身心健康，为他们创造良好的生活和学习环境。只有这样，才能让母爱在我们的生命中发挥出更大的价值和意义。
</td> </tr>
</table>

经过我们的微调，目前 TechGPT-2.0 对其不知道的问题会直接进行不可回答的回复，对与人类价值观相悖的问题会进行否定回答并给出合理建议。

另外，还需注意，上述结果并非设置为相同的输出长度。在默认输出长度设置下，模型输出的结果偏保守，即很多正确的结果在强制大模型输出更长结果的条件下可以得到，但在默认输出长度设置条件下，这些正确的结果并没有被模型输出；有时也可能因为强制模型输出的长度过长，导致模型过分填充一些无用字符。

## 总结
和 TechGPT-1.0 相比，**TechGPT-2.0** 在许多方面进行了改进，新模型具有更强的**知识图谱构建**能力、以及**智能问答**能力。尤其是在**嵌套NER**以及**医学、法律**领域能力的提升，并且模型对**幻觉等边界问题以及长文本问题**的回答也得到了改善。

目前，我们对TechGPT的研究也在持续进行中，如外挂知识库、大模型与知识图谱融合、长文本问题优化等方向。

## 主要贡献者
排名不分先后
<table>
  <tr>
    <td align='center'>
      <img src="https://avatars.githubusercontent.com/u/86044648?v=4" alt="Contributor 1" height="150">
      <br>
      <b>wangjiaqi @ Northeastern University</b>
      <br>
        <a href='https://github.com/wangjiaqi886'>TechGPT-2.0 项目的核心开发成员</a>
    </td>
    <td align='center'>
      <img src="https://avatars.githubusercontent.com/u/109963333?v=4" alt="Contributor 2" height="150">
      <br>
      <b>Chang Yuying @ Northeastern University</b>
      <br>
        <a href='https://github.com/changyuying'>TechGPT-2.0 项目的核心开发成员</a>
    </td>
    <td align='center'>
      <img src="https://avatars.githubusercontent.com/u/49021250?v=4" alt="Contributor 3" height="150">
      <br>
      <b>SIGMOID @ Northeastern University</b>
      <br>
        <a href='https://github.com/lizhongv'>TechGPT-2.0 项目的核心开发成员</a>
    </td>
  </tr>
  </table>


## 免责声明

该项目仅供学习交流使用，禁止用于商业用途。在使用过程中，使用者需认真阅读并遵守以下声明:

1. 本项目仅为大模型测试功能而生，使用者需自行承担风险和责任，如因使用不当而导致的任何损失或伤害，本项目概不负责。
2. 本项目中出现的第三方链接或库仅为提供便利而存在，其内容和观点与本项目无关。使用者在使用时需自行辨别，本项目不承担任何连带责任；
3. 使用者在测试和使用模型时，应遵守相关法律法规，如因使用不当而造成损失的，本项目不承担责任，使用者应自行承担；若项目出现任何错误，请向我方反馈，以助于我们及时修复；
4. 本模型中出现的任何违反法律法规或公序良俗的回答，均不代表本项目观点和立场，我们将不断完善模型回答以使其更符合社会伦理和道德规范。

使用本项目即表示您已经仔细阅读、理解并同意遵守以上免责声明。本项目保留在不预先通知任何人的情况下修改本声明的权利。

## 引用

如果使用本项目的代码、数据或模型，请引用本项目。

```
@misc{TechGPT,
  author = {Feiliang Ren and Jiaqi Wang and Yuying Chang and Zhong Li},
  title = {TechGPT 2.0: Technology-Oriented Generative Pretrained Transformer 2.0},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/neukg/TechGPT-2.0}},
}
```

## 致谢

**我们对 [Chinese-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) 和 [Atom](https://github.com/FlagAlpha/Llama2-Chinese) 的相关项目和研究开发人员表示衷心的感谢！**

```
@article{Chinese-LLaMA-Alpaca,
    title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca},
    author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
    journal={arXiv preprint arXiv:2304.08177},
    url={https://arxiv.org/abs/2304.08177},
    year={2023}
}
```

