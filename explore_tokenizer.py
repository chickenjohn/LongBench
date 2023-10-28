from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from datasets import load_dataset
from pred import load_model_and_tokenizer, build_chat
from matplotlib import pyplot as plt
import numpy as np
import torch, json, os

model2path = json.load(open("config/model2path.json", "r"))
model2maxlen = json.load(open("config/model2maxlen.json", "r"))
dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model_name = "chatglm2-6b-32k"
# define your model
model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
max_length = model2maxlen[model_name]


datasets = list(dataset2maxlen.keys())
datasets = ["hotpotqa"]
print(datasets)
for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]

    all_context_len = []

    for json_obj in data:
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        all_context_len.append(context_length)

    print(f"dataset {dataset} mean context len: {np.mean(all_context_len)}")

    plt.figure()
    plt.hist(all_context_len, 20)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.ylabel("count")
    plt.xlabel("input ids len")
    plt.savefig(f"context-len-hist-{model_name}-{dataset}.png")