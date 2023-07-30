# from datasets import load_dataset,load_from_disk
# import json
# downloaded = load_dataset('yahma/alpaca-cleaned')
# downloaded.save_to_disk('/data/home/tingchenfu/AlpacaLora/temp')
# reloaded = load_from_disk('/data/home/tingchenfu/AlpacaLora/temp')


# f=open('/data/home/tingchenfu/AlpacaLora/temp/train.jsonl','w')
# for i in range(len(reloaded['train'])):
#     f.write(json.dumps({'instruction': reloaded['train'][i]['instruction'],
#                         'input':reloaded['train'][i]['input'],
#                         'output':reloaded['train'][i]['output'], 
#                         #'text': reloaded['train'][i]['text']
#                         },ensure_ascii=False)+'\n')
    

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from utils.prompter import Prompter
from  model.tokenization_llama import LlamaTokenizer
from transformers import default_data_collator
model_name_or_path='/apdcephfs/share_916081/tingchenfu/PLM/llama-7b'
prompter = Prompter('alpaca')
train_file = '/apdcephfs/share_916081/tingchenfu/Dataset/Alpaca/train.jsonl'
streaming = False
window_size = 128
loss_on_input = False


tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
tokenizer.padding_side = "left" 
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
    
    # print(len(input_id_list))
    # print(input_id_list[0])
    return {
        'input_id': torch.tensor(input_id_list,dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_list,dtype = torch.long),
        'label': torch.tensor(label_list,dtype=torch.long),
    }


dataset = load_dataset("json", data_files = train_file, streaming=streaming)
print(dataset['train'])
train_dataset= dataset['train'].map(
    preprocess_fn,
    fn_kwargs={'loss_on_input': loss_on_input, 'window_size': window_size},
    batched=True,
    #num_proc= None
)

train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=False,collate_fn = default_data_collator)

for step,epoch in enumerate(train_dataloader):
    pass
print('here is ok')