from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              trust_remote_code=True)
    
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({
                                    'pad_token': '<pad>',
                                    'bos_token':'<bos>', 
                                    'eos_token': '<eos>'}
                                 )
    
    return model, tokenizer