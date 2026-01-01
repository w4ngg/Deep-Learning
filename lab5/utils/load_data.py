import json

def load_viodc_dataset(json_path, tokenizer, label_field="domain"):
    """
    Load UIT-ViOCD dataset for classification

    Args:
        json_path: path to train.json
        tokenizer: WordTokenizer
        label_field: "domain" or "label"

    Returns:
        encodings, labels, label2id, id2label
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    raw_labels = []

    for sample in data.values():
        texts.append(sample["review"])
        raw_labels.append(sample[label_field])

    # build label mapping
    unique_labels = sorted(set(raw_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    labels = [label2id[l] for l in raw_labels]

    # tokenize
    encodings = {
        "input_ids": [],
        "attention_mask": []
    }

    for text in texts:
        item = tokenizer.encode(text)
        encodings["input_ids"].append(item["input_ids"])
        encodings["attention_mask"].append(item["attention_mask"])

    return encodings, labels, label2id, id2label

def load_phoner_dataset(json_path, tokenizer):
    """
    Load PhoNER dataset from JSON lines format

    Each line:
    {
      "words": [...],
      "tags":  [...]
    }

    Returns:
        encodings: dict(input_ids, attention_mask)
        tag_ids: List[List[int]]
        tag2id: Dict[str, int]
    """

    sentences = []
    tags = []

    # ---- Read file ----
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sentences.append(item["words"])
            tags.append(item["tags"])

    # ---- Build tag vocab ----
    unique_tags = sorted({t for seq in tags for t in seq})
    tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    encodings = {
        "input_ids": [],
        "attention_mask": []
    }
    tag_ids = []

    max_len = tokenizer.max_len

    # ---- Encode each sentence ----
    for words, tag_seq in zip(sentences, tags):
        input_ids = [tokenizer.cls_id]
        label_ids = [-100]   # ignore CLS

        for w, t in zip(words, tag_seq):
            wid = tokenizer.word2id.get(w.lower(), tokenizer.unk_id)
            input_ids.append(wid)
            label_ids.append(tag2id[t])

        input_ids.append(tokenizer.sep_id)
        label_ids.append(-100)  # ignore SEP

        # truncate
        input_ids = input_ids[:max_len]
        label_ids = label_ids[:max_len]

        attention_mask = [1] * len(input_ids)

        # padding
        while len(input_ids) < max_len:
            input_ids.append(tokenizer.pad_id)
            attention_mask.append(0)
            label_ids.append(-100)

        encodings["input_ids"].append(input_ids)
        encodings["attention_mask"].append(attention_mask)
        tag_ids.append(label_ids)

    return encodings, tag_ids, tag2id, id2tag

def load_viodc_dataset_with_mapping(
    json_path,
    tokenizer,
    label2id,
    label_field="domain"
):
    """
    Load dev/test for classification using label2id from TRAIN
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    encodings = {
        "input_ids": [],
        "attention_mask": []
    }
    labels = []

    for sample in data.values():
        text = sample["review"]
        label = sample[label_field]

        if label not in label2id:
            raise ValueError(f"Unknown label in dev/test: {label}")

        item = tokenizer.encode(text)

        encodings["input_ids"].append(item["input_ids"])
        encodings["attention_mask"].append(item["attention_mask"])
        labels.append(label2id[label])

    return encodings, labels


def load_phoner_dataset_json_with_mapping(json_path, tokenizer, tag2id):
    """
    Load PhoNER dev/test (JSON Lines format)
    Each line is a JSON object with keys: 'words', 'tags'
    """

    encodings = {
        "input_ids": [],
        "attention_mask": []
    }
    tag_ids = []
    max_len = tokenizer.max_len

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue   # bỏ dòng trống

            item = json.loads(line)

            # 🔥 VALIDATE FORMAT (tránh KeyError)
            if "words" not in item or "tags" not in item:
                continue
            if len(item["words"]) != len(item["tags"]):
                continue

            words = item["words"]
            tags = item["tags"]

            input_ids = [tokenizer.cls_id]
            label_ids = [-100]

            for w, t in zip(words, tags):
                wid = tokenizer.word2id.get(w.lower(), tokenizer.unk_id)

                if t not in tag2id:
                    raise ValueError(f"Unknown tag in dev/test: {t}")

                input_ids.append(wid)
                label_ids.append(tag2id[t])

            input_ids.append(tokenizer.sep_id)
            label_ids.append(-100)

            # truncate
            input_ids = input_ids[:max_len]
            label_ids = label_ids[:max_len]

            attention_mask = [1] * len(input_ids)

            # padding
            while len(input_ids) < max_len:
                input_ids.append(tokenizer.pad_id)
                attention_mask.append(0)
                label_ids.append(-100)

            encodings["input_ids"].append(input_ids)
            encodings["attention_mask"].append(attention_mask)
            tag_ids.append(label_ids)

    if len(tag_ids) == 0:
        raise RuntimeError("No valid PhoNER samples loaded")

    return encodings, tag_ids
