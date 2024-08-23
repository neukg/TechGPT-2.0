import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# 加载模型和分词器
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    'your path',
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    # trust_remote_code=True,
    # use_flash_attention_2=True
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    'your path', use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# 定义生成函数


def generate_response(input_text, max_new_tokens, min_new_tokens, top_k, top_p, temperature, repetition_penalty, do_sample, num_beams):
    prompt = f"<s>Human: {input_text}\n</s><s>Assistant: "
    input_ids = tokenizer(prompt, return_tensors="pt",
                          add_special_tokens=False).input_ids
    input_ids = input_ids.to(model.device)

    generate_input = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "num_beams": num_beams,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }
    generate_ids = model.generate(**generate_input)
    text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    text = text.split("Assistant:")[-1]
    return text


# 使用 gr.Blocks 创建界面
with gr.Blocks() as demo:
    gr.Markdown("# 文本生成应用")
    gr.Markdown("使用 Atom-7B 模型生成文本。")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                lines=5, placeholder="输入提示文本...", label="提示")
            max_new_tokens = gr.Slider(
                10, 1024, value=512, step=1, label="最大新生成的 tokens 数")
            min_new_tokens = gr.Slider(
                0, 100, value=0, step=1, label="最小生成的 tokens 数")
            top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-P")
            temperature = gr.Slider(
                0.0, 2.0, value=1.0, step=0.01, label="Temperature")
            repetition_penalty = gr.Slider(
                1.0, 2.0, value=1.3, step=0.1, label="重复惩罚")
            do_sample = gr.Checkbox(label="使用采样生成 (do_sample)", value=False)
            num_beams = gr.Slider(1, 10, value=1, step=1,
                                  label="束搜索数 (num_beams)")

        with gr.Column():
            output_text = gr.Textbox(label="生成文本")

    generate_button = gr.Button("生成文本")

    # 按钮点击事件
    generate_button.click(
        fn=generate_response,
        inputs=[prompt_input, max_new_tokens, min_new_tokens, top_k,
                top_p, temperature, repetition_penalty, do_sample, num_beams],
        outputs=output_text
    )

# 启动 Gradio 接口
demo.launch(server_name="0.0.0.0", server_port=31814)
