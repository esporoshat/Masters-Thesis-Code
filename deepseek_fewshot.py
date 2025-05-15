import pandas as pd
import torch
import re
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, accuracy_score

# --- Load DeepSeek Model and Tokenizer ---
model_name = "deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.config.pad_token_id = model.config.eos_token_id

# --- Load ESG Dataset ---
train_df = pd.read_csv("train_augmented_esg.csv")
test_df  = pd.read_csv("test_esg.csv")

LABELS    = ["Environment", "Social", "Governance"]
TEXT_COL  = "Text"
LABEL_COL = "ESG_Category"

# --- Sample Few‐Shot (2 per class) ---
few_shot = []
for lab in LABELS:
    few_shot.append(
        train_df[train_df[LABEL_COL] == lab]
        .sample(2, random_state=42)
    )
few_shot_df = pd.concat(few_shot)

# --- Precompute force_words_ids for our 3 labels ---
force_words_ids = [
    [tokenizer(lab, add_special_tokens=False).input_ids[0]]
    for lab in LABELS
]

# --- System Instruction to Force One‐Word Answer ---
system_msg = (
    "You are an expert ESG classifier. "
    "When given a piece of text, reply with exactly one word: "
    "Environment, Social, or Governance."
)

# --- Build Prompt String ---
def build_prompt(text):
    prompt  = f"SYSTEM: {system_msg}\n\n"
    for _, r in few_shot_df.iterrows():
        prompt += (
            "USER: Classify the ESG category of the following text:\n"
            f"{r[TEXT_COL]}\n"
            f"ASSISTANT: {r[LABEL_COL]}\n\n"
        )
    prompt += (
        "USER: Classify the ESG category of the following text:\n"
        f"{text}\n"
        "ASSISTANT:"
    )
    return prompt

# --- Generate & Extract with forced labels ---
preds = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
    # 1) Build prompt & tokenize
    prompt = build_prompt(row[TEXT_COL])
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(model.device)

    # 2) Generate exactly 1 token from our label set
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        num_beams=3,
        do_sample=False,
        pad_token_id=model.config.eos_token_id,
        force_words_ids=force_words_ids,
        return_dict_in_generate=True,
        output_scores=True
    )

    # 3) Decode and map back to full label
    gen_id = outputs.sequences[0][-1]
    gen = tokenizer.decode(gen_id).strip().lower()
    
    # 4) Robust mapping with fallback to logits
    if any(x in gen for x in ["environment", "env"]):
        preds.append("Environment")
    elif any(x in gen for x in ["social", "soc"]):
        preds.append("Social")
    elif any(x in gen for x in ["governance", "gov"]):
        preds.append("Governance")
    else:
        # Fallback: Use logits to choose most probable valid label
        logits = outputs.scores[0][0]
        valid_label_indices = [force[0] for force in force_words_ids]
        best_idx = torch.argmax(logits[valid_label_indices]).item()
        preds.append(LABELS[best_idx])

# --- Sanity check lengths ---
assert len(preds) == len(test_df), f"Got {len(preds)} preds for {len(test_df)} samples"

# --- Save Predictions ---
results_df = pd.DataFrame({
    "Text":       test_df[TEXT_COL],
    "True_Label": test_df[LABEL_COL],
    "Predicted":  preds
})
results_df.to_csv("deepseek_preds.csv", index=False)
print("Predictions saved to deepseek_preds.csv")


# --- Evaluation ---
report = classification_report(test_df[LABEL_COL], preds, labels=LABELS, output_dict=True)
accuracy    = accuracy_score(test_df[LABEL_COL], preds)
macro_f1    = report["macro avg"]["f1-score"]
# micro_f1    = report["micro avg"]["f1-score"]  # REMOVE THIS LINE
weighted_f1 = report["weighted avg"]["f1-score"]

print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro F1‐Score:    {macro_f1:.4f}")
# print(f"Micro F1‐Score:    {micro_f1:.4f}")    # REMOVE THIS LINE
print(f"Weighted F1‐Score: {weighted_f1:.4f}\n")
print("Full Classification Report:")
print(classification_report(test_df[LABEL_COL], preds, labels=LABELS))
