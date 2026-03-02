import runpod
import sys
import os

print("=== HANDLER STARTING ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)

# Phase 1: Just test that the worker runs at all
def handler(job):
    print(f"=== GOT JOB ===", flush=True)
    inp = job["input"]
    mode = inp.get("mode", "test")

    if mode == "test":
        return {"status": "alive", "message": "Worker is running!"}

    # Phase 2: Load model on demand (only when mode=paraphrase)
    if mode == "paraphrase":
        print("Importing torch...", flush=True)
        import torch
        print("Importing transformers...", flush=True)
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import nltk
        nltk.download("punkt_tab", quiet=True)

        print("Loading tokenizer...", flush=True)
        tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        print("Loading model...", flush=True)
        model = T5ForConditionalGeneration.from_pretrained(
            "kalpeshk2011/dipper-paraphraser-xxl",
            torch_dtype=torch.float16,
        ).to("cuda")
        model.eval()
        print("Model loaded!", flush=True)

        text = inp.get("text", "")
        lex = inp.get("lex_diversity", 40)
        order = inp.get("order_diversity", 40)
        top_p = inp.get("top_p", 0.75)

        sentences = nltk.sent_tokenize(text)
        output_text = ""
        for start in range(0, len(sentences), 3):
            window = " ".join(sentences[start:start + 3])
            lex_code = int(100 - lex)
            order_code = int(100 - order)
            input_text = f"lexical = {lex_code}, order = {order_code}"
            if output_text:
                input_text += f" {output_text}"
            input_text += f" <sent> {window} </sent>"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, do_sample=True, top_p=top_p, max_length=512)
            result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            output_text += " " + result

        return {"paraphrased": output_text.strip()}

    return {"error": f"Unknown mode: {mode}"}

print("=== STARTING RUNPOD HANDLER ===", flush=True)
runpod.serverless.start({"handler": handler})
