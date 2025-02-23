from unsloth import FastLanguageModel

def initialize_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./qwen25_32b_instruct/cwe",
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


