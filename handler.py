import runpod
import torch
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer

nltk.download("punkt_tab", quiet=True)

# Load model once at cold start
print("Loading DIPPER model...")
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
model = T5ForConditionalGeneration.from_pretrained(
    "kalpeshk2011/dipper-paraphraser-xxl",
    torch_dtype=torch.float16,
).to("cuda")
model.eval()
print("Model loaded.")


def paraphrase(text, lex_diversity=40, order_diversity=40, prefix="", do_sample=True, top_p=0.75, max_length=512):
    """Paraphrase text using DIPPER. Diversity values are SIMILARITY codes (40 = 60% diversity)."""
    sentences = nltk.sent_tokenize(text)
    output_text = ""
    interval = 3  # process 3 sentences at a time

    for start in range(0, len(sentences), interval):
        window = " ".join(sentences[start:start + interval])

        # Build control codes
        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)
        input_text = f"lexical = {lex_code}, order = {order_code}"

        # Add context prefix
        if prefix:
            input_text += f" {prefix}"
        if output_text:
            input_text += f" {output_text}"

        input_text += f" <sent> {window} </sent>"

        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=do_sample,
                top_p=top_p,
                max_length=max_length,
            )

        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output_text += " " + result

    return output_text.strip()


def handler(job):
    inp = job["input"]
    text = inp.get("text", "")
    if not text:
        return {"error": "No text provided"}

    lex = inp.get("lex_diversity", 40)
    order = inp.get("order_diversity", 40)
    prefix = inp.get("prefix", "")
    do_sample = inp.get("do_sample", True)
    top_p = inp.get("top_p", 0.75)

    result = paraphrase(text, lex_diversity=lex, order_diversity=order, prefix=prefix, do_sample=do_sample, top_p=top_p)
    return {"paraphrased": result}


runpod.serverless.start({"handler": handler})
