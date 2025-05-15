import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== CONFIG ====================
MODEL_PATH = "saved_model"  # Adjust if needed
TEST_PATH = "test_esg.csv"
TEXT_COL = "Text"
LABEL_COL = "ESG_Category"
LABELS = ["Environment", "Social", "Governance"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== LOAD MODEL ====================
print("Loading ESG-LLaMA...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# ==================== CLASSIFY FUNCTION ====================
def classify_llama(text):
    prompt = (
        "Instruction: Classify the following text into one of these ESG categories: "
        "Environment, Social, or Governance.\n"
        f"Input: {text}\n"
        "Output:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            do_sample=False
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"\n[DEBUG] Prompt: {prompt}")
    print(f"[DEBUG] LLaMA Response: {generated_text}")

    # Try to extract label
    match = re.search(r'(Environment|Social|Governance)', generated_text, re.IGNORECASE)
    if not match:
        print(f"[WARNING] Could not extract label from: {generated_text}")
        return "None"
    
    return match.group(1).capitalize()

# ==================== MAIN ====================
def main():
    print("Reading test dataset...")
    df = pd.read_csv(TEST_PATH)
    texts = df[TEXT_COL].tolist()
    true_labels = df[LABEL_COL].tolist()

    print("Running zero-shot ESG classification with ESG-LLaMA...\n")
    predictions = []

    for text in texts:
        label = classify_llama(text)
        predictions.append(label)

    df["ESG_LLaMA_Prediction"] = predictions

    output_file = "esg_llama_zero_shot_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n Predictions saved to: {output_file}")

if __name__ == "__main__":
    main()
