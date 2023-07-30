import os
import sys
from typing import List
import argparse
from str2bool import str2bool
import datetime
import logging
import torch
from torch.utils.data import DataLoader
import transformers
from datasets import load_dataset
from tqdm.auto import tqdm
import math

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs
)
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler


from transformers import (
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from model.configuration_llama import LlamaConfig
from model.modeling_llama import LlamaForCausalLM
from model.tokenization_llama import LlamaTokenizer
from utils.prompter import Prompter

MOUNT_DIR = '/apdcephfs/share_916081/shared_info/tingchenfu'
RUN_DIR='/'.join(os.path.abspath(__file__).split('/')[:-2])


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--debug",type=str2bool,default=True)
    parser.add_argument("--model_name_or_path", type=str,default='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/llama-7b-hf')
    parser.add_argument("--tokenizer_name", type=str, default='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/llama-7b-hf', help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--config_name",type=str,default='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/llama-7b-hf',help="Pretrained tokenizer name or path if not the same as model_name",)
    
    parser.add_argument("--train_file",type=str,default='/apdcephfs/share_916081/shared_info/tingchenfu/AlpacaLora/train.jsonl')
    parser.add_argument("--streaming",type=str2bool,default=False,help="whether using streaming dataset")
    parser.add_argument(
        "--micro_train_bs",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--mirco_eval_bs",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--accum_step",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--n_epoch", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_step",
        type=int,
        default=100000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--n_warmup_step", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=512,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=100,
        help="use accelerate print to print out some training dynamics"
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=100,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--with_tracking",
        type=str2bool,
        default=False,

    )
    parser.add_argument("--special_name",type=str,default=None)
    parser.add_argument("--special_setting",type=str,default=None)
    parser.add_argument("--output_dir",type=str)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str2bool,
        default=False,
        help="If the training should continue from a checkpoint folder.",
    )
    
    # special parameter in this file
    parser.add_argument("--prompt_template_name",type=str,default='alpaca')
    parser.add_argument("--lora_r",type=int,default=8)
    parser.add_argument("--lora_alpha",type=int,default=16)
    parser.add_argument("--lora_dropout",type=float,default=0.05)
    parser.add_argument("--lora_target_module",type=str,default='q_proj,k_proj,v_proj,o_proj')
    
    
    parser.add_argument("--loss_on_input",type=str2bool,default=True,help='whether the input has loss')


    parser.add_argument("--load_8bit",type=str2bool,default=True)

    args = parser.parse_args()



    args.lora_target_module = args.lora_target_module.split(',')
    if args.debug:
        args.output_dir=os.path.join(MOUNT_DIR,'AlpacaLora/dump','debug')
        args.micro_train_bs = 2
        args.accum_step=1
        args.print_step=1
    else:
        exp_name = '{}_{}'.format(args.model_name_or_path.split('/')[-1], args.train_file.split('/')[-2])
        if args.special_name:
            exp_name = args.special_name+'_' + exp_name
        exp_setting = 'seq' + str(args.window_size) + 'bs' + str(args.micro_train_bs *torch.cuda.device_count()*int(os.environ['HOST_NUM'])*args.accum_step) + 'lr' + str(args.lr) + 'warm'+ str(args.n_warmup_step)+args.lr_scheduler_type
        if args.special_setting:
            exp_setting = args.special_setting+'_'+ exp_setting
        args.output_dir = os.path.join(RUN_DIR,'dump',exp_name, exp_setting)

    os.makedirs(args.output_dir,exist_ok = True)
    return args


# def train(
#     # model/data params
#     base_model: str = "",  # the only required argument
#     data_path: str = "/apdcephfs/share_916081/tingchenfu/Dataset/Alpaca/train.jsonl",
#     output_dir: str = "./lora-alpaca",
#     # training hyperparams
#     batch_size: int = 128,
#     micro_batch_size: int = 4,
#     num_epochs: int = 3,
#     learning_rate: float = 3e-4,
#     cutoff_len: int = 256,
#     val_set_size: int = 2000,
#     # lora hyperparams
    
#     # llm hyperparams
#     loss_on_inputs: bool = True,  # if False, masks out inputs in loss
#     group_by_length: bool = False,  # faster, but produces an odd training loss curve
#     # wandb params
#     wandb_project: str = "",
#     wandb_run_name: str = "",
#     wandb_watch: str = "",  # options: false | gradients | all
#     wandb_log_model: str = "",  # options: false | true
#     resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
#     prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
# ):
#     pass
    
def main():
    #torch.dynamo.config.verbose=True
    args = parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.accum_step,
                            log_with = 'wandb',
                            project_dir=args.output_dir,
                            kwargs_handlers= [InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=3600*10))],
                        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    #logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "log"), 'w'))
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=True)
    for k,v in vars(args).items():
        logger.info("{}= {}".format(k,v))
    accelerator.print(f"{AcceleratorState()}")


    prompter = Prompter(args.prompt_template_name)

    if args.tokenizer_name is not None:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name)
    else:
        assert args.model_name_or_path is not None
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'':torch.cuda.current_device()}
    )   

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.load_8bit:
        model = prepare_model_for_int8_training(model)
    
    
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_module,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # model=torch.compile(model)

    if accelerator.is_main_process:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    model.config.use_cache = False


    if torch.__version__ >= "2" and sys.platform != "win32":
        accelerator.print('compile!!!')
        model = torch.compile(model,backend='aot_eager')

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))


    def preprocess_fn(examples,loss_on_input,window_size):
        instruction_list = examples['instruction']
        input_list = examples['input']
        output_list = examples['output']
        bs = len(input_list)
        
        input_id_list=[]
        attention_mask_list=[]
        label_list=[]
        
        for i in range(bs):
            raw = prompter.generate_prompt(instruction_list[i], input_list[i])
            input_id = tokenizer.encode(raw,truncation = False, padding=False)
            if loss_on_input:
                label = [-100]*len(input_id)
            else:
                label = input_id+[]
            input_id += tokenizer.encode(output_list[i]) + [ tokenizer.eos_token_id ]
            label += tokenizer.encode(output_list[i]) + [ tokenizer.eos_token_id ]
            attention_mask = [1] * len(input_id)
            
            assert len(input_id) == len(attention_mask) == len(label)

            input_id_list.append(input_id)
            attention_mask_list.append(attention_mask)
            label_list.append(label)


        for i in range(bs):
            input_id_list[i] = input_id_list[i][:window_size]
            attention_mask_list[i] = attention_mask_list[i][:window_size]
            label_list[i] = label_list[i][:window_size]

            if len(input_id_list[i]) < window_size:
                padding_length = window_size - len(input_id_list[i])
                input_id_list[i].extend([tokenizer.pad_token_id] * padding_length)
                attention_mask_list[i].extend([0] * padding_length)
                label_list[i].extend([-100] * padding_length)

        return {
            'input_ids': torch.tensor(input_id_list,dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_list,dtype = torch.long),
            'labels': torch.tensor(label_list,dtype=torch.long),
        }


    dataset = load_dataset("json", data_files = args.train_file, streaming=args.streaming)
    # if args.debug:
    #     accelerator.print("train dataset truncation")
    #     train_dataset = dataset['train'].select(list(range(128)))
    # else:
    #     train_dataset = dataset['train']
    train_dataset = dataset['train']

    with accelerator.main_process_first():
        train_dataset= train_dataset.map(
            preprocess_fn,
            fn_kwargs={'loss_on_input': args.loss_on_input, 'window_size': args.window_size},
            batched=True,
            #num_proc= None
        )
    accelerator.wait_for_everyone()

    train_dataloader = DataLoader(train_dataset,batch_size=args.micro_train_bs,shuffle=True,collate_fn=default_data_collator)

    # no_decay = ["bias", "layer_norm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]

    optimizer_cls = (torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)


    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name = args.lr_scheduler_type,
            optimizer = optimizer,
            num_warmup_steps = args.n_warmup_step,
            num_training_steps = args.max_train_step,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_step, warmup_num_steps=args.n_warmup_step
        )

    # Prepare everything with our `accelerator`.
    # model = accelerator.prepare_model(model)
    # optimizer = accelerator.prepare_optimizer(optimizer)
    # lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader,  lr_scheduler
    )

    if not args.streaming:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.accum_step)
        args.max_train_step = args.n_epoch * num_update_steps_per_epoch

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.


    # Figure out how many steps we should save the Accelerator states
    # checkpoint_step = args.checkpoint_step
    # if checkpoint_step is not None and checkpoint_step.isdigit():
    #     checkpoint_step = int(checkpoint_step)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(
            project_name=args.exp_name, 
            config=experiment_config,
            )
        

    completed_step=0
    resume_step = None
    if args.resume_from_checkpoint:
        for file in os.listdir(args.output_dir):
            if 'step_ckpt' in file:
                resume_step= int(file.replace('step_ckpt',''))
                break
        if resume_step is None:
            accelerator.print('checkpoint not found and could not be resumed')
        else:
            
            checkpoint_path = os.path.join(args.output_dir,file)
            accelerator.load_state(checkpoint_path)
            accelerator.print('load from {}'.format(checkpoint_path))

        
        
        
        # Check the available weights and load them
        # checkpoint_name = os.path.join(
        #     resume_from_checkpoint, "pytorch_model.bin"
        # )  # Full checkpoint
        # if not os.path.exists(checkpoint_name):
        #     checkpoint_name = os.path.join(
        #         resume_from_checkpoint, "adapter_model.bin"
        #     )  # only LoRA model - LoRA config above has to fit
        #     resume_from_checkpoint = (
        #         False  # So the trainer won't try loading its state
        #     )
        # # The two files above have a different name depending on how they were saved, but are actually the same.
        # if os.path.exists(checkpoint_name):
        #     print(f"Restarting from {checkpoint_name}")
        #     adapters_weights = torch.load(checkpoint_name)
        #     model = set_peft_model_state_dict(model, adapters_weights)
        # else:
        #     print(f"Checkpoint {checkpoint_name} not found")
    




    accelerator.print("start training!!!")
    total_batch_size = args.micro_train_bs * accelerator.num_processes * args.accum_step
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.n_epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.micro_train_bs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.accum_step}")
    logger.info(f"  Total optimization steps = {args.max_train_step}")
    progress_bar = tqdm(range(args.max_train_step), disable=not accelerator.is_local_main_process)


    for epoch in range(args.n_epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            total_loss = 0
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and resume_step is not None and not args.streaming and completed_step < resume_step:
                if step % args.accum_step == 0:
                    progress_bar.update(1)
                    completed_step += 1
                continue

            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            # if args.with_tracking:
            total_loss += loss.detach().float()
            loss = loss / args.accum_step
            accelerator.backward(loss)
            if (step + 1) % args.accum_step == 0 or (not args.streaming and step == len(train_dataloader) - 1):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # Checks if the accelerator has performed an optimization step behind the scenes
                # if accelerator.sync_gradients:
                if not args.streaming:
                    progress_bar.update(1)
                completed_step += 1

                if not args.streaming and completed_step >= args.max_train_step:
                    break

                if args.with_tracking :
                    accelerator.log(
                        {
                            "train_loss": total_loss.item()/completed_step,
                            "epoch": epoch,
                            "step": completed_step,
                        },
                        step=completed_step,
                    )
            
                if completed_step and completed_step  %  args.print_step ==0:
                    accelerator.print("epoch: {} completed_step {} training loss {} ".format(epoch,completed_step,total_loss/completed_step))

                if completed_step>0 and completed_step % args.checkpoint_step == 0 and not args.streaming:
                    accelerator.wait_for_everyone()
                    accelerator.save_state(output_dir = os.path.join(args.output_dir,'{}step_ckpt'.format(completed_step)))
                    for file in os.listdir(args.output_dir):
                        if 'step_ckpt' in file and str(completed_step) not in file:
                            os.system('rm -rf '+ os.path.join(args.output_dir,file))

                # plan 1:
                # plan 2:
                # state_dict=accelerator.unwrap_model(model).state_dict()
                # accelerator.save(state_dict, os.path.join(args.output_dir,'{}step_model'.format(completed_step)))
                # plan 3:
                # success = model.save_checkpoint(args.output_dir, "{}step_ckpt".format(completed_step), {'epoch':epoch,'last_global_step':completed_step})
                # status_msg = f"checkpointing: checkpoint_folder={args.output_dir}, ckpt_id={completed_step}"
                # if success:
                #     logging.info(f"Success {status_msg}")
                # else:
                #     logging.warning(f"Failure {status_msg}")
        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir = os.path.join(args.output_dir,'{}epoch_ckpt'.format(epoch)))


if __name__ == "__main__":
    main()