from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import torch

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset (without cleaning)
file_path = "sasb_esg.csv"  # Use the dataset without text processing
df = pd.read_csv(file_path, encoding="utf-8")

# Load pre-trained T5 model and tokenizer on the correct device
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)  # Move model to device
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

def paraphrase_text(text, num_return_sequences=1):
    """Generates paraphrased versions of the input text using T5 while ensuring all tensors are on the same device."""
    max_length = min(len(text.split()) * 2, 100)  # Set max length dynamically

    input_text = f"paraphrase: {text}"
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt", truncation=True, max_length=max_length)

    # Move tensors to the same device as the model
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    return [tokenizer.decode(o, skip_special_tokens=True) for o in output]

# Filter only the "Environment" category for augmentation
env_df = df[df["Main Category"] == "Environment"]

# Number of paraphrases per sample
num_paraphrases = 1  

# Generate paraphrased versions
augmented_data = []
for _, row in env_df.iterrows():
    for _ in range(num_paraphrases):
        paraphrased_text = paraphrase_text(row["Text"], num_return_sequences=1)[0]
        augmented_data.append({
            "Text": paraphrased_text,  # Store the new paraphrased version
            "Main Category": row["Main Category"]  # Keep the category label unchanged
        })

# Convert augmented data to DataFrame
augmented_df = pd.DataFrame(augmented_data)

# Combine the original dataset with the augmented dataset
balanced_df = pd.concat([df, augmented_df], ignore_index=True)

# Save the final dataset
balanced_file_path = "train_augmented.csv"
balanced_df.to_csv(balanced_file_path, index=False, encoding="utf-8")

print(f"Augmented dataset saved as: {balanced_file_path}")
