from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              trust_remote_code=True)
    
    tokenizer.add_special_tokens({
                                  'bos_token': '<s>', 
                                  'eos_token': '<|endoftext|>',
                                  'pad_token': '<|endoftext|>',
                                  'unk_token': '<unk>'}
                                 )
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    return model, tokenizer