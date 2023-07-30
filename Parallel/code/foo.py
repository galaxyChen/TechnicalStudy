# import torch
# import transformers
# from peft import PeftModel
# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# weight_path='/apdcephfs/share_916081/shared_info/tingchenfu/work1/dump/debug/0epoch_ckpt/pytorch_model.bin'
# weight = torch.load(weight_path)
# print(weight.keys())
# base_model='/apdcephfs/share_916081/tingchenfu/PLM/llama-7b-hf'
# lora_weights='/apdcephfs/share_916081/tingchenfu/AlpacaLora/dump/reimpl'

# model = LlamaForCausalLM.from_pretrained(
#     base_model,
#     load_in_8bit=False,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
# model = PeftModel.from_pretrained(
#     model,
#     lora_weights,
#     torch_dtype=torch.float16,
# )
# print('here is ok')

############################################################################################
# dataset length statistics
# import json
# from tqdm import tqdm
# import numpy as np
# tokenizer=LlamaTokenizer.from_pretrained('/apdcephfs/share_916081/shared_info/tingchenfu/PLM/llama-7b-hf')
# f=open('/apdcephfs/share_916081/shared_info/tingchenfu/AlpacaLora/train.jsonl')
# length=[]
# for line in tqdm(f.readlines()):
#     data=json.loads(line)
#     text=data['instruction']+data['input']+data['output']
#     encoded=tokenizer.encode(text,max_length=512,truncation=True)
#     length.append(len(encoded))

# print(sum(length)/len(length))
# print(np.sum(np.array(length)>512))
# print(np.sum(np.array(length)>256))
# print(np.sum(np.array(length)>128))
# print(np.sum(np.array(length)>64))


############################################################################################
# import torch
# import transformers
# from peft import PeftModel
# # from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# reloaded = torch.load('/apdcephfs/share_916081/shared_info/tingchenfu/work1/dump/peft_sentalign_bloom-7b/seq1024bs128lr0.001warm500cosine/4100step_ckpt/adapter_model.bin')
# print(reloaded.keys())


######################################################################################################
from transformers import AutoTokenizer

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from utils.prompter import Prompter
from transformers import default_data_collator


prompter = Prompter('alpaca')
model_name_or_path = '/apdcephfs/share_916081/shared_info/tingchenfu/PLM/gpt2-large'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.add_special_tokens(special_tokens_dict={'pad_token':'[PAD]'})
train_file = '/apdcephfs/share_916081/shared_info/tingchenfu/AlpacaLora/train.jsonl'
micro_train_bs=2


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


dataset = load_dataset("json", data_files = [train_file], streaming=False)
train_dataset = dataset['train']

train_dataset= train_dataset.map(
    preprocess_fn,
    fn_kwargs={'loss_on_input': True, 'window_size': 512},
    batched=True,
    #num_proc= None
)
train_dataloader = DataLoader(train_dataset,batch_size=micro_train_bs,shuffle=True,collate_fn=default_data_collator)


for step, batch in enumerate(train_dataloader):
    #print(tokenizer.decode(batch['input_ids'][0],skip_special_tokens=True))
    #print(tokenizer.decode(batch['labels'][0],skip_special_tokens=True))
    print((batch['input_ids'][0][:128]))
    print((batch['labels'][0][:128]))
    break
