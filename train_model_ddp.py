
# # Reference: 
# - https://www.youtube.com/watch?v=gXDsVcY8TXQ&t=3146s
# - https://github.com/TrelisResearch/install-guides/blob/main/multi-gpu/test_scripts/test_pp.py
# - https://github.com/huggingface/trl/issues/1303


import os
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["HF_HOME"] = "/NS/llm-1/nobackup/afkhan/HF_CACHE/Misc"
os.environ["HF_DATASETS_CACHE"] = "/NS/llm-1/nobackup/afkhan/HF_CACHE/Datasets"
os.environ["TRANSFORMERS_CACHE"] = "/NS/llm-1/nobackup/afkhan/HF_CACHE/Models"


cache_dir = os.getenv("TRANSFORMERS_CACHE")


# !pip install transformers datasets bitsandbytes deepspeed accelerate

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import PartialState 

# !pip install wandb
import wandb
from utils import print_trainable_parameters


## Parallelism Related


DEVICE_MAP = 'DDP'

if DEVICE_MAP == "DDP":
    DEVICE_STRING = PartialState().process_index
    DEVICE_MAP={'':DEVICE_STRING}
    print("Device Map for DDP: ", DEVICE_MAP)


## Load Model


model_name = 'Gemma-2B'
model_path = "/NS/llm-1/nobackup/vnanda/llm_base_models/gemma-2b"

model = AutoModelForCausalLM.from_pretrained(
    model_path, cache_dir=cache_dir, device_map=DEVICE_MAP
)

tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token


## Load Dataset and Preprocess

ds_name = "weyaxi--sci-datasets"
ds = load_dataset("Weyaxi/sci-datasets", "alpaca")


# Keep only 100 examples for now
ds['train'] = ds['train'].select(range(100))


def merge_columns(example):
    example['text'] = '### Instruction: ' + example['instruction'] + ' ### Answer: ' + example['output']
    return example


ds['train'] = ds['train'].map(merge_columns)


ds['train'][0]


ds = ds.map(
    lambda samples: tokenizer(samples["text"]), batched=True,
)


## Setting Hyperparams


## Wandb Related

WANDB_PROJECT = "FSDP-Analysis"
WANDB_RUN_NAME = f"{model_name}-{ds_name}-full-finetune" + "-ddp"

## Logging Related

REPORT_TO = "wandb"
OUTPUT_DIR = f"./output/{model_name}-{ds_name}-full-finetune" + "-ddp"
LOGGING_DIR = f"./logs/{model_name}-{ds_name}-full-finetune" + "-ddp"
LOGGING_STRATEGY = "steps"
LOGGING_STEPS = 10

## Training Duration Related

MAX_STEPS = 1000

## Optimizer Related

LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "linear"
WARMUP_RATIO = 0.1

## Batch Related

PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8

## Gradient Related (Also related to Parallelism)

GRADIENT_CHECKPOINTING = True
# Use reentrant starts a more efficient method of recomputing the graph from checkpoints
USE_REENTRANT = False # Set False for DDP and True for Model/Pipeline Parallelism


training_args = TrainingArguments(
    # Logging Related
    report_to=REPORT_TO,
    output_dir = OUTPUT_DIR,
    logging_dir = LOGGING_DIR,
    logging_strategy = LOGGING_STRATEGY,
    logging_steps = LOGGING_STEPS,
    # Training Duration Related
    max_steps = MAX_STEPS,
    # Optimizer Related
    learning_rate = LEARNING_RATE,
    lr_scheduler_type = LR_SCHEDULER_TYPE,
    warmup_ratio = WARMUP_RATIO,
    # Batch Related
    per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size = PER_DEVICE_EVAL_BATCH_SIZE,
    # Gradient Related
    gradient_checkpointing = GRADIENT_CHECKPOINTING,
    gradient_checkpointing_kwargs = {"use_reentrant": USE_REENTRANT},
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=ds['train'],
    eval_dataset=ds['train'],
)


# Configure Wandb project and run

wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
wandb.config.update(training_args)


trainer.train()


# Save Model
model.save_pretrained(f'Saves/{model_name}-{ds_name}-full-finetune' + "-ddp")


