import sys
import os
import webbrowser
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

app = Flask(__name__, template_folder=resource_path('templates'))

def translate_english_to_hebrew(text):
    model_dir = resource_path('models/nllb-200-distilled-600M')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    
    src_lang = "eng_Latn"
    tgt_lang = "heb_Hebr"
    
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        english_text = request.form['english_text']
        translated_text = translate_english_to_hebrew(english_text)
        return render_template('translate.html', translated_text=translated_text)
    return render_template('translate.html', translated_text=None)

if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=False)
