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
                                  
                                    'pad_token': '<|padding|>',
                                    'bos_token': '<s>', 
                                    'eos_token': '</s>'
                                  }
                                 )
    tokenizer.padding_side = 'right'

    return model, tokenizer