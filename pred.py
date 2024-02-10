import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import shortuuid as su
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--dataset', nargs='*', default=[])
    parser.add_argument('--local_rank', type=int, default=0, help="rank number")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--deepspeed', action='store_true', help="Enable Deepspeed Inference")
    parser.add_argument('--num_insts', type=int, default=-1, help="#instances to use")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()        
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, deepspeed=False, num_insts=-1):
    preds = []
    insts_counter = 0
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        print(f"curr_len: {context_length}")
        if context_length > 5000:
            continue
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            if deepspeed:
                import deepspeed
                ds_engine = deepspeed.init_inference(
                    model,
                    mp_size=1,
                    dtype=torch.half,
                    checkpoint=None,
                    replace_with_kernel_inject=True,
                    zero = {
                        "stage": 3,
                        "offload_param": {
                            "device": "cpu",
                            "pin_memory": True
                        },
                        "overlap_comm": False
                    }
                )
                model = ds_engine.module

            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
        insts_counter += 1
        if num_insts > 0 and insts_counter == num_insts:
            break

    return preds

def get_pred_sliced(model, tokenizer, data, max_length, max_gen, prompt_format_struct, prompt_format_compos, dataset, device, model_name, deepspeed=False, num_insts=-1):
    preds = []
    insts_counter = 0
    for json_obj in tqdm(data):
        data_all_comps = {**json_obj, **prompt_format_compos}
        # get lengths of the components
        compos_len = {k: len(tokenizer(data_all_comps[k], truncation=False, return_tensors="pt").input_ids[0]) \
                      for k in prompt_format_struct["involved_members"]}
        # compute start and end of components
        compos_pos, curr_end = {}, -1
        for k in compos_len.keys():
            compos_start = curr_end + 1
            compos_end = compos_start + compos_len[k] - 1
            compos_pos[k] = (compos_start, compos_end)
            curr_end = compos_end
        prompt = prompt_format_struct["struct"].format(**data_all_comps)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        print(f"curr_len: {context_length}")
        if context_length > 5000:
            continue
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            if deepspeed:
                import deepspeed
                ds_engine = deepspeed.init_inference(
                    model,
                    mp_size=1,
                    dtype=torch.half,
                    checkpoint=None,
                    replace_with_kernel_inject=True,
                    zero = {
                        "stage": 3,
                        "offload_param": {
                            "device": "cpu",
                            "pin_memory": True
                        },
                        "overlap_comm": False
                    }
                )
                model = ds_engine.module

            # create a random inst id and save slice info
            attn_path = "/chronos_data/tji/.huggingface_cache/transformers/chatglm2-6b-32k-attn-bfp20-1e-3-hotpotqa-bprune-scaled/"
            inst_id = su.random(length=8)
            with open(attn_path + f"i{inst_id}.json", "w") as f:
                json.dump(compos_pos, f)
                
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                # attn_dump_path=attn_path + f"i{inst_id}",
                attn_dump_path=None,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
        insts_counter += 1
        if num_insts > 0 and insts_counter == num_insts:
            break

    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)   
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        print("loading chatglm")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    max_length = model2maxlen[model_name]
    if args.dataset:
        datasets = args.dataset
    elif args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    dataset2prompt_comps = json.load(open("config/dataset2prompts_sliced_comps.json", "r"))
    dataset2prompt_struct = json.load(open("config/dataset2prompts_sliced_struc.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        # preds = get_pred(model, 
        #                  tokenizer, 
        #                  data, 
        #                  max_length, 
        #                  max_gen, 
        #                  prompt_format, 
        #                  dataset, 
        #                  device, 
        #                  model_name, 
        #                  args.deepspeed,
        #                  args.num_insts)
        preds = get_pred_sliced(model, 
                                tokenizer, 
                                data, 
                                max_length, 
                                max_gen, 
                                dataset2prompt_struct[dataset],
                                dataset2prompt_comps[dataset], 
                                dataset, 
                                device, 
                                model_name, 
                                args.deepspeed,
                                args.num_insts)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')