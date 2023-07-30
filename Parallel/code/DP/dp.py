import os
import sys
import argparse
from str2bool import str2bool
import datetime
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
import math
from transformers import (
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
# from transformers import LlamaConfig,LlamaTokenizer,LlamaForCausalLM
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from utils.prompter import Prompter
# modifier 1
from torch.nn.parallel import DataParallel

MOUNT_DIR = '/apdcephfs/share_916081/shared_info/tingchenfu'
RUN_DIR='/'.join(os.path.abspath(__file__).split('/')[:-2])


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--debug",type=str2bool,default=True)
    parser.add_argument("--model_name_or_path", type=str,default='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/bloom-560m')
    parser.add_argument("--tokenizer_name", type=str, default='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/bloom-560m', help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--config_name",type=str,default='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/bloom-560m',help="Pretrained tokenizer name or path if not the same as model_name",)
    
    parser.add_argument("--train_file",type=str,default='/apdcephfs/share_916081/shared_info/tingchenfu/AlpacaLora/train.jsonl')
    parser.add_argument("--streaming",type=str2bool,default=False,help="whether using streaming dataset")
    
    parser.add_argument(
        "--micro_train_bs",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--mirco_eval_bs",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--accum_step",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--clip",type=float,default=2.0)
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
    
    # special parameter in this file
    parser.add_argument("--prompt_template_name",type=str,default='alpaca')
    parser.add_argument("--lora_r",type=int,default=8)
    parser.add_argument("--lora_alpha",type=int,default=16)
    parser.add_argument("--lora_dropout",type=float,default=0.05)
    parser.add_argument("--lora_target_module",type=str,default='q_proj,k_proj,v_proj,o_proj')
    parser.add_argument("--loss_on_input",type=str2bool,default=False,help='whether the input has loss')
    parser.add_argument("--load_8bit",type=str2bool,default=False)

    args = parser.parse_args()



    args.lora_target_module = args.lora_target_module.split(',')
    if args.debug:
        args.output_dir=os.path.join(MOUNT_DIR,'AlpacaLora/dump','debug')
        args.micro_train_bs = 8
        args.accum_step=4
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

    
def main():
    #torch.dynamo.config.verbose=True
    args = parse_args()
    for k,v in vars(args).items():
        print("{} == {}".format(k,v))
    device=torch.device('cuda:0')
    prompter = Prompter(args.prompt_template_name)

    if args.tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        assert args.model_name_or_path is not None
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    #tokenizer.padding_side = "left"  # Allow batched inference

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        #load_in_8bit=args.load_8bit,
        #torch_dtype=torch.float16,
        # device_map='auto',
        #device_map={'':torch.cuda.current_device()}
    )
    print("model loaded")

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    model.to(device)
    model  = DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
    

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
                label = input_id+[]
            else:
                label = [-100]*len(input_id)
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
    train_dataset = dataset['train']
    train_dataset= train_dataset.map(
        preprocess_fn,
        fn_kwargs={'loss_on_input': args.loss_on_input, 'window_size': args.window_size},
        batched=True,
        #num_proc= None
    )
    train_dataloader = DataLoader(train_dataset,batch_size=args.micro_train_bs,shuffle=True,collate_fn=default_data_collator)

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.lr)

    lr_scheduler = get_scheduler(
        name = args.lr_scheduler_type,
        optimizer = optimizer,
        num_warmup_steps = args.n_warmup_step,
        num_training_steps = args.max_train_step,
    )


    if not args.streaming:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.accum_step)
        args.max_train_step = args.n_epoch * num_update_steps_per_epoch
        progress_bar = tqdm(range(args.max_train_step))

    completed_step=0


    for epoch in range(args.n_epoch):
        model.train()
        total_loss=0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            outputs = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device), 
                            labels=batch['labels'].to(device),
                        )
            loss = torch.mean(outputs.loss,dim=0)
            
            total_loss += loss.detach().float()
            loss = loss / args.accum_step
            loss.backward()
            if (step + 1) % args.accum_step == 0 or (not args.streaming and step == len(train_dataloader) - 1):
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
                if grad_norm >= 1e2:
                    print('WARNING : Exploding Poly Gradients {:.2f}'.format(grad_norm))
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
                if completed_step and completed_step  %  args.print_step ==0:
                    print("epoch: {} completed_step {} training loss {} ".format(epoch,completed_step,total_loss))
                if completed_step>0 and completed_step % args.checkpoint_step == 0 and not args.streaming:
                    torch.save(model.module.state_dict(), os.path.join(args.output_dir,'{}step_ckpt'.format(completed_step)))
                    for file in os.listdir(args.output_dir):
                        if 'step_ckpt' in file and str(completed_step) not in file:
                            os.system('rm -rf '+ os.path.join(args.output_dir,file))
                
                total_loss=0

if __name__ == "__main__":
    main()