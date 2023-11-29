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
    temperature=0.3,
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

example = '出血因凝血因子大量被消耗、血小板减少及继发纤溶亢进，发生出血。\n请列示这段文本中的所有特定实体。'

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