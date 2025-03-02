# %%
import json
import re
import random
import zipfile
import os 
import ast
from transformers import pipeline , GPTNeoForCausalLM, GPT2Tokenizer
from PyPDF2 import PdfReader
from datasets import Dataset
# %%
def extract_text_from_pdf(zip_path):
    extracted_texts = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.pdf'):
                pdf_data = zip_ref.read(file_name)
                from io import BytesIO
                pdf_stream = BytesIO(pdf_data)
                reader = PdfReader(pdf_stream)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                extracted_texts.append(text)
    return extracted_texts
# %%
def parse_problems(text):
    problems = re.split(r'Problem \d+:', text)[1:]
    return [p.strip() for p in problems]
# %%
# Initialize model with error handling
try:
    model_name = "EleutherAI/gpt-neo-1.3B"
    generator = pipeline("text-generation", model=model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    raise
# %%
def annotate_problems_with_llm(problem_text):
    prompt = ("Extract the following components from the given math problem:\n"
              "- Entities\n- Relationships\n- Constraints\n- Goal\n\n"
              f"Problem: {problem_text}\n\nReturn the response in JSON format.")
    response = generator(prompt, max_length=500, do_samples = True, temperature = 0.7)
    response_text = response[0]['generated_text'].strip()

    try:
        parsed_output = ast.literal_eval(response_text) 
    except (SyntaxError, ValueError):
        parsed_output = {"entities": [], "relationships": [], "constraints": [], "goal": ""}
    parsed_output["text"] = problem_text
    return parsed_output      
# %%
def split_dataset(annotated_problems):
    random.shuffle(annotated_problems)
    total = len(annotated_problems)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    train_set = annotated_problems[:train_size]
    val_set = annotated_problems[train_size:train_size + val_size]
    test_set = annotated_problems[train_size + val_size:]

    return train_set, val_set, test_set
# %%
def prepare_finetune_dataset(annotated_problems):
    dataset = Dataset.from_dict({
        "input": [p["text"] for p in annotated_problems],
        "output": [json.dumps({"entities": p["entities"], "relationships": p["relationships"],
                                "constraints": p["constraints"], "goal": p["goal"]}) for p in annotated_problems]
    })
    return dataset
# %%
if __name__ == "__main__":
    zip_path = r"C:\Users\HP\Desktop\My Documents\CompCodes\AIMO_Cide\Problems_CIDE.zip"
    texts = extract_text_from_pdf(zip_path)
    problems = [p for text in texts for p in parse_problems(text)]
    annotated_problems = [annotate_problems_with_llm(p) for p in problems]
    train_set, val_set, test_set = split_dataset(annotated_problems)

    dataset = prepare_finetune_dataset(annotated_problems)

    dataset.save_to_disk("imo_finetune_dataset")

    with open("imo_dataset.json", "w") as f:
        json.dump({"train": train_set, "val": val_set, "test": test_set}, f, indent=4)

    print("Ho gya dataset")
# %%
