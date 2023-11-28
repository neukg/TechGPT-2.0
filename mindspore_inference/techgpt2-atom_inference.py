import os
import argparse
import numpy as np
import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool


TEMPLATE = (
    "<s>Human: "
    "{instruction} \n</s><s>Assistant: "
)


def generate_prompt(instruction):
    return TEMPLATE.format_map({'instruction': instruction})

def context_init(use_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


def main(model_type='llama_7b',
         use_parallel=False,
         device_id=0,
         checkpoint_path="",
         use_past=True):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)

    # 多batch输入
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

    # set model config
    model_config = AutoConfig.from_pretrained(model_type)
    model_config.use_past = use_past
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    # build model from config
    network = AutoModel.from_config(model_config)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(os.getenv("RANK_ID", "0")))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard pangualpha and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    for index, example in enumerate(inputs):
        inputs[index] = generate_prompt(instruction=example)
    text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer)
    outputs = text_generation_pipeline(inputs)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama2_7b', type=str,
                        help='which model to use.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='./target_checkpoint/rank_0/llama2_7b0.ckpt', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    args = parser.parse_args()

    main(args.model_type,
         args.use_parallel,
         args.device_id,
         args.checkpoint_path,
         args.use_past)