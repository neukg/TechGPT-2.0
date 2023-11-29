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
    top_p=0.95,
    top_k=50,
    num_beams=1,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=2,
    max_new_tokens=1024,
    min_new_tokens=10,
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