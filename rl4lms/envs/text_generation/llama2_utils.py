import json
from transformers import LlamaTokenizer

def load_data(data_path,prompt_path=None):
    # .json file
    with open(data_path,'r') as f:
        data = json.load(f)

    # .txt file
    if prompt_path:
        with open(prompt_path,'r',encoding='utf-8') as f:
            prompts = f.read().strip()
        return data,prompts
    return data

def load_model_tokenizer(model_path):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    model.cuda()
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)  # 报protoc版本错误
    except:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = 'right'  # right和left都可以
    tokenizer.eos_token = '\n\n'
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token = tokenizer.eos_token
    return model, tokenizer

def llama_reader(model_path,data_path,max_length,eval_batch_size,prompt_path,topk=1,output_path=None,save_results=False):
    # 1.data preparing:
    data, prompts =load_data(data_path,prompt_path)

    # 2.model preparing:
    model, tokenizer = load_model_tokenizer(model_path)
    # ?不太懂
    collator_function = data_proc.Collator_(maxlength=max_length, tokenizer=tokenizer)
    eval_dataset = data_proc.Dataset_(data)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
        # num_workers=1,
        collate_fn=collator_function,
        drop_last=False,
    )

    # 3. predicting answers with llama model:
