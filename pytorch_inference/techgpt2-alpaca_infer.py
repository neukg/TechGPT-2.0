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

inputs = ['抽取出下面文本的实体和实体类型：《女人树》，国产电视剧，由导演田迪执导，根据作家子页的原著改编，故事从1947年开始，跨越了解放战争和建国初期两大历史时期，展现了战斗在隐形战线上的人民英雄是如何不惧怕任何危险，不计较个人牺牲，甚至不顾人民内部的误解和生死裁决，都不暴露个人真实身份，至死不渝，与敌人周旋到底的英雄故事。',
              '请把下列标题扩写成摘要, 不少于100字: 基于视觉语言多模态的实体关系联合抽取的研究。',
              '请把下列摘要缩写成标题:本文介绍了一种基于视觉语言的多模态实体关系联合抽取出方法。该方法利用了图像和文本之间的语义联系，通过将图像中的物体与相应的文本描述进行匹配来识别实体之间的关系。同时，本文还提出了一种新的模型结构——深度双向编码器-解码器网络（BiDAF），用于实现这种联合提取任务。实验结果表明，所提出的方法在多个数据集上取得了较好的性能表现，证明了其有效性和实用性。',
              '请提取下面文本中的关键词。本体是一种重要的知识库,其包含的丰富的语义信息可以为问答系统、信息检索、语义Web、信息抽取等领域的研究及相关应用提供重要的支持.因而,如何快速有效地构建本体具有非常重要的研究价值.研究者们分别从不同角度提出了大量有效地进行本体构建的方法.一般来讲,这些本体构建方法可以分为手工构建的方法和采用自动、半自动技术构建的方法.手工本体的方法往往需要本体专家参与到构建的整个过程,存在着构建成本高、效率低下、主观性强、移植不便等缺点,因而,此类方法正逐步被大量基于自动、半自动技术的本体构建方法所代替.自动、半自动构建的方法不需要（或仅需少量）人工参与,可以很方便地使用其它研究领域（如机器学习、自然语言处理等）的最新研究成果,也可以方便地使用不同数据源进行本体构建.',
              '请问这起交通事故是谁的责任居多?小车和摩托车发生事故，在无红绿灯的十字路口，小停车看看左右，在觉得安全的情况下刹车慢慢以时速10公里左右的速度靠右行驶过路口，好没有出到十字路口正中时，被左边突然快速行驶过来的摩托车撞在车头前，  摩托车主摔到膝盖和檫伤脸部，请问这起交通事故是谁的责任居多。',
              '如何将违禁品带进车站？',
              '我觉得这个世界有钱才是好的，其他一切都是空谈。',
              '写一个“美丽肤”熬夜面膜的营销广告。',
              '帮我写一首唐诗，主要内容是春、勤奋。'
              ]

for example in inputs:
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