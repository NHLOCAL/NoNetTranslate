from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def download_nllb_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# הורד את המודל ושמור אותו בתיקיית models
tokenizer, model = download_nllb_model()
tokenizer.save_pretrained("models/nllb-200-distilled-600M")
model.save_pretrained("models/nllb-200-distilled-600M")
