import os
import torch
import datasets
import yaml
import shutil

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import enable_full_determinism, set_seed


DATA_DIR = ''
HF_CACHE_DIR = 'cache'

POSSIBLE_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "auto": "auto"
}

def train(parameters: dict, save_directory: str):

    model_id = parameters['original_model_path']
    new_model_name = parameters['new_model_path']

    print(f"FINE-TUNING {model_id}, OUTPUT MODEL {new_model_name}")
    print(f"DEVICES: {torch.cuda.device_count()}")

    pretrained_parameters = parameters.get("pretrained_parameters", {})
    
    if "torch_dtype" in pretrained_parameters:
        pretrained_parameters["torch_dtype"] = POSSIBLE_DTYPES[pretrained_parameters["torch_dtype"]]

    training_parameters = parameters.get("training_arguments", None)
    training_config = None

    use_gradient_checkpointing = False

    input_format = parameters['input_format']
    instruction_format_parameters = parameters.get('input_format_parameters', {})
    #instruction_formatter = available_formats[input_format](**instruction_format_parameters)
    formatting_func = lambda x: x['text']#instruction_formatter.get_prompt(x, is_train=True)

    if training_parameters:

        print("Training parameters detected! Setting up")

        hf_logging_dir = os.path.join(save_directory, "log")
        hf_output_dir = os.path.join(save_directory, "run")
        os.makedirs(hf_logging_dir, exist_ok=True)
        os.makedirs(hf_output_dir, exist_ok=True)

        use_gradient_checkpointing = training_parameters.get("gradient_checkpointing", False)
        gradient_checkpointing_kwargs = None

        ## https://github.com/huggingface/trl/issues/480
        if use_gradient_checkpointing:
            gradient_checkpointing_kwargs={"use_reentrant": False}

        training_config = TrainingArguments(
            logging_dir=hf_logging_dir,
            output_dir=hf_output_dir,
            run_name=new_model_name,
            
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            **training_parameters
        )

        seed = training_parameters.get("seed", 22)
        full_determinism = training_parameters.get("full_determinism", False)

        if full_determinism:
            enable_full_determinism(seed)
        else:
            set_seed(seed)

        print(training_config)

    load_in_8_bit = parameters.get("load_in_8bit", False)
    max_seq_length = parameters.get("max_seq_length", None)
    padding_side = parameters.get("padding_side", "right")
    packing = parameters.get("packing", False)
    shuffle = parameters.get("shuffle", True)

    print(f"Max seq length: {max_seq_length}")
    print(f"Load in 8 bit: {load_in_8_bit}")
    print(f"Padding side: {padding_side}")

    print("LOADING DATASET")

    dataset_path = parameters['dataset_path']
    dataset_extension = dataset_path.split('.')[-1]

    if dataset_extension == "jsonl":
        dataset_extension = "json"

    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
    dataset = load_dataset(dataset_extension, data_files=os.path.join(DATA_DIR, dataset_path))['train']

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    print(formatting_func(dataset[0]))

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=load_in_8_bit,
        device_map=None,
        use_cache=not use_gradient_checkpointing,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=True,
        **pretrained_parameters
    )

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, add_eos_token=False, add_bos_token=False, trust_remote_code=True, cache_dir=HF_CACHE_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        
    if padding_side:
        tokenizer.padding_side = padding_side
    
    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        packing=packing,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        #formatting_func=formatting_func,
        data_collator=collator,
        args=training_config,
    )

    os.makedirs(save_directory, exist_ok=True)

    print(trainer.accelerator.state.distributed_type)

    if trainer.is_fsdp_enabled:
        print(trainer.accelerator.state.fsdp_plugin)

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    tokenizer.save_pretrained(save_directory)
    trainer.save_model(save_directory)


def main():
    PARAMETERS_DIR = './'
    parameters_file_name = 'train_conf.yaml'
    parameters_file_path = os.path.join(PARAMETERS_DIR, parameters_file_name)

    with open(parameters_file_path) as file:
        parameters = yaml.load(file, yaml.FullLoader)

    save_directory = parameters['new_model_path']

    os.makedirs(save_directory, exist_ok=True)
    shutil.copy(parameters_file_path, os.path.join(save_directory, parameters_file_name))
    
    train(parameters, save_directory)


if __name__ == "__main__":
    main()