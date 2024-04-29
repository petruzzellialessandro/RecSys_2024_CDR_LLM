from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os 
import pickle
from tqdm import tqdm
import json
import yaml
from datasets import load_dataset, disable_caching
from transformers.trainer_utils import set_seed
import numpy as np


POSSIBLE_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "auto": "auto"
}



def inference(parameters_inf: dict, model_path: str, dataset_path: str, output_dir: str, dataset_setting: str):
    pretrained_parameters = parameters_inf.get("pretrained_parameters", {})
    generation_parameters = parameters_inf.get("generation_parameters", {})
    batch_size = parameters_inf.get("batch_size", 8)
    seed = parameters_inf.get('seed', 22)
    model_name_high = parameters_inf.get('model_name_high', 'llama-chat')
    
    dataset_extension = 'json'
    if "torch_dtype" in pretrained_parameters:
            pretrained_parameters["torch_dtype"] = POSSIBLE_DTYPES[pretrained_parameters["torch_dtype"]]
        
    if "torch_dtype" in generation_parameters:
        generation_parameters["torch_dtype"] = POSSIBLE_DTYPES[generation_parameters["torch_dtype"]]
    set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=None,
            device_map="auto",
            cache_dir='cache',
            **pretrained_parameters
        )
    model.eval()
    model = model.bfloat16()
    padding_side = parameters_inf["padding_side"]
    #generation_parameters['padding_side'] = padding_side
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, add_eos_token=False, add_bos_token=False, trust_remote_code=True, cache_dir="cache")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    settings_file = [x for x in os.listdir(dataset_path) if dataset_setting in x]
    for dataset_name_conf in settings_file:
        dataset = load_dataset(dataset_extension, data_files=os.path.join(dataset_path, dataset_name_conf))['train']
        print(dataset[0])
        field = 'text'
        accelerator = Accelerator()
        results = {
            'outputs':[],
            'inputs':[]
        }
        
        accelerator.wait_for_everyone()

        progress_bar = tqdm(range(len(dataset)), disable=not accelerator.is_main_process)

        with accelerator.split_between_processes(dataset[field]) as inp:
            results=dict(outputs=[], inputs=[])
            with torch.no_grad():

                for i in range(0, len(inp), batch_size):

                    batch_inp = inp[i:i+batch_size]
                    results['inputs'].extend(inp[i:i+batch_size])
                    prompt_tokenized=tokenizer(batch_inp, return_tensors="pt", padding=True).to("cuda")
                    output_tokenized = model.generate(**prompt_tokenized, **generation_parameters)
                    # remove prompt from output 
                    # remove prompt from gen. tokens
                    outputs_tokenized=[ tok_out[len(tok_in):] 
                        for tok_in, tok_out in zip(prompt_tokenized["input_ids"], output_tokenized) ] 
                    outputs=tokenizer.batch_decode(outputs_tokenized)
                    results["outputs"].extend(outputs)
                    # store outputs and number of tokens in result{}
                    #results['outputs'].append(tokenizer.batch(output_tokenized) )
                    progress_bar.update(batch_size)
                results=[ results ] 
        # Wait for all processes to finish
        accelerator.wait_for_everyone()
        # Gather the results
        results = gather_object(results)
        with accelerator.main_process_first():
            os.makedirs(output_dir, exist_ok=True)
            filehandler = open(os.path.join(output_dir, f'{model_name_high}-{dataset_name_conf}.pkl'), 'wb') 
            pickle.dump(results, filehandler)
            processed_ds_path = os.path.join(output_dir, f'{model_name_high}-{dataset_name_conf}.jsonl')

            with open(processed_ds_path, 'w', encoding='utf8') as f_out:

                for x, result in zip(dataset, results):

                        x[field] = result

                        json.dump(x, f_out)
                        f_out.write('\n')


def main():
    PARAMETERS_DIR = "ale_priv/CDR/inferenceCDR/"
    parameters_file_name = 'config.yaml'
    parameters_file_path = os.path.join(PARAMETERS_DIR, parameters_file_name)

    with open(parameters_file_path) as file:
        parameters_inf = yaml.load(file, yaml.FullLoader)

    model_dir = parameters_inf["model_dir"]
    dataset_path = parameters_inf["dataset_path"]
    dataset_setting = parameters_inf["dataset_setting"]
    output_dir = parameters_inf["output_dir"]

    parameters_path = os.path.join(model_dir, 'parameters.yaml')

    if os.path.isfile(parameters_path):
        with open(parameters_path) as file:
            parameters = yaml.load(file, yaml.FullLoader)
    else:
        parameters = {}

    inference(parameters_inf, model_dir, dataset_path, output_dir, dataset_setting)


if __name__ == "__main__":
    main()