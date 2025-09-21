import os
import re
import json
import shutil
import tempfile
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from fuzzywuzzy import fuzz

# ==============================
# 1. Config
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "distilbert-base-uncased")
JSON_PATH = os.path.join(BASE_DIR, "..", "data", "pokemon_commands.json")
AUGMENTED_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "pokemon_commands_augmented.json")
MODEL_NAME = "distilbert-base-uncased"

pokemon_to_id = {"pikachu": 0, "charizard": 1, "bulbasaur": 2, "mewtwo": 3}

# Aliases for fuzzy matching
pokemon_aliases = {
    "pikachu": ["pikachu", "yellow electric rodent", "shock mouse","lightning mouse"],
    "charizard": ["charizard", "fire lizard","fire dragon"],
    "bulbasaur": ["bulbasaur", "green seed dinosaur"],
    "mewtwo": ["mewtwo", "psychic clone"]
}

# Attack/protect verbs for augmentation
attack_verbs = ["kill", "neutralize", "take down", "eliminate"]
protect_verbs = ["avoid harming", "protect", "keep safe", "do not attack"]

# BERT label mapping
bert_label_map = {0: "attack", 1: "protect"}

# Minimum similarity threshold for fuzzy matching
FUZZY_THRESHOLD = 70  # 0-100 scale

# ==============================
# 2. JSON augmentation
# ==============================
def augment_json(original_path, augmented_path):
    if os.path.exists(original_path):
        with open(original_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_texts = set(d["text"].lower() for d in existing_data)
    augmented_data = []

    # Include all existing JSON sentences/aliases
    for entry in existing_data:
        text = entry["text"].strip()
        pokemon = entry["pokemon"].lower()
        label = entry["label"]
        if text.lower() not in existing_texts:
            augmented_data.append({"text": text, "label": label, "pokemon": pokemon})
            existing_texts.add(text.lower())

    # Generate extra attack/protect sentences for each alias
    for pokemon in pokemon_to_id.keys():
        for alias in pokemon_aliases.get(pokemon, []):
            for verb in attack_verbs:
                sentence = f"{verb} {alias}"
                if sentence.lower() not in existing_texts:
                    augmented_data.append({"text": sentence, "label": "attack", "pokemon": pokemon})
                    existing_texts.add(sentence.lower())
            for verb in protect_verbs:
                sentence = f"{verb} {alias}"
                if sentence.lower() not in existing_texts:
                    augmented_data.append({"text": sentence, "label": "protect", "pokemon": pokemon})
                    existing_texts.add(sentence.lower())

    combined_data = existing_data + augmented_data

    with open(augmented_path, "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"[INFO] JSON updated. {len(augmented_data)} new entries added.")
    print(f"[INFO] Updated JSON saved to: {augmented_path}")
    return combined_data

# ==============================
# 3. Download base model
# ==============================
def download_base_model():
    if not os.path.exists(MODEL_DIR):
        print(f"[INFO] Base model not found locally. Downloading {MODEL_NAME}...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)
        print(f"[INFO] Base model downloaded and saved to {MODEL_DIR}")
    else:
        print(f"[INFO] Base model already exists in {MODEL_DIR}")

# ==============================
# 4. Train BERT
# ==============================
def train_bert(json_data):
    download_base_model()
    print(f"[INFO] Training BERT model...")

    label_map = {"attack": 0, "protect": 1}
    dataset = Dataset.from_list([
        {"text": d["text"], "label": label_map[d["label"]]}
        for d in json_data
    ])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    dataset = dataset.map(lambda batch: tokenizer(batch["text"], padding=True, truncation=True), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=2, local_files_only=True
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_strategy="epoch",
        logging_dir=os.path.join(BASE_DIR, "..", "logs"),
        logging_steps=50
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    # Windows-safe save
    temp_dir = tempfile.mkdtemp(prefix="bert_model_")
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    shutil.move(temp_dir, MODEL_DIR)

    print(f"[INFO] Model trained and saved to {MODEL_DIR}")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    return classifier

# ==============================
# 5. Fuzzy match Pokémon names
# ==============================
def match_pokemon(sentence):
    matched_ids = set()
    for name, aliases in pokemon_aliases.items():
        for alias in aliases:
            similarity = fuzz.partial_ratio(alias.lower(), sentence.lower())
            if similarity >= FUZZY_THRESHOLD:
                matched_ids.add(pokemon_to_id[name])
    return matched_ids

# ==============================
# 6. Parse paragraph
# ==============================
def parse_paragraph(paragraph, classifier, threshold=0.5):
    sentences = re.split(r'[.!?]', paragraph)
    targets, protected = set(), set()

    sentences_clean = [s.strip() for s in sentences if s.strip()]
    if not sentences_clean:
        return [], []

    results = classifier(sentences_clean, batch_size=8)

    for sentence, result in zip(sentences_clean, results):
        label_int = int(result["label"].split("_")[1])
        label = bert_label_map.get(label_int, None)
        score = result.get("score", 0)
        if label is None or score < threshold:
            continue

        matched_ids = match_pokemon(sentence)
        if label == "attack":
            targets.update(matched_ids)
        elif label == "protect":
            protected.update(matched_ids)

    return list(targets), list(protected)

# ==============================
# 7. Main
# ==============================
if __name__ == "__main__":
    combined_data = augment_json(JSON_PATH, AUGMENTED_JSON_PATH)
    classifier = train_bert(combined_data)

    paragraph = input("Enter paragraph with kill/protect commands: ")
    targets, protected = parse_paragraph(paragraph, classifier)
    print(f"Targets: {targets}, Protected: {protected}")
