# The Visual StoryTeller — Instructions

This project implements an image captioning system using:

- **EfficientNet-B4** — Encoder
- **Transformer Decoder** — Caption Generator

All inference logic is already implemented inside the notebook. No code modification is required.

---

## 1. Kaggle Setup

1. Create a new Kaggle Notebook
2. Enable GPU (recommended)

---

## 2. Required Inputs

### Dataset

Upload `captions_data.zip`, containing:

- Image dataset
- Caption metadata used during inference

### Model Files

Create a Kaggle Dataset named **`neededFiles`**, and upload *all* of the following files from the provided `model` folder:

| File | Description |
|---|---|
| `config.json` | Model configuration |
| `encoder_best.pt` | Best encoder checkpoint |
| `encoder_best_bleu.pt` | Encoder checkpoint, best BLEU score |
| `encoder_final.pt` | Final encoder checkpoint |
| `decoder_best.pt` | Best decoder checkpoint |
| `decoder_best_bleu.pt` | Decoder checkpoint, best BLEU score |
| `decoder_final.pt` | Final decoder checkpoint |
| `test_captions.csv` | Held-out test captions |
| `vocab.pkl` | Vocabulary object |

**Important:** Do not rename any file or modify the folder structure. All filenames must match exactly.

---

## 3. Attach Inputs in Kaggle

After uploading, click **Add Input** and attach:

- `captions_data.zip`
- the `model` folder (i.e. `neededFiles`)

Both will be mounted automatically under `/kaggle/input/`.

---

## 4. Run the Inference Notebook

Run all cells from top to bottom. The notebook will automatically:

1. Load model configuration (`config.json`)
2. Load the vocabulary (`vocab.pkl`)
3. Load encoder and decoder weights
4. Process the input images
5. Generate captions using beam search

---

## 5. Expected Kaggle Directory Structure

```
/kaggle/input/
│
├── captions-data/
│   ├── images/
│   └── captions.csv
│
└── neededFiles/
    ├── config.json
    ├── encoder_best.pt
    ├── encoder_best_bleu.pt
    ├── encoder_final.pt
    ├── decoder_best.pt
    ├── decoder_best_bleu.pt
    ├── decoder_final.pt
    ├── vocab.pkl
    └── test_captions.csv
```

---

## 6. Inference Pipeline

The model works in four steps:

**1. Load Encoder**
EfficientNet-B4 extracts image features.

**2. Feature Extraction**
Image → high-dimensional feature vectors.

**3. Transformer Decoder**
Processes features together with previously generated words.

**4. Caption Generation**
Outputs a natural-language sentence using beam search.

---

## 7. Key Notes

- Internet is not required — all weights are local
- GPU is recommended for faster inference
- Ensure all files are uploaded before running
- Do not change filenames or folder structure

---

## 8. Output Example

The model generates captions such as:

- A man riding a bicycle on the street
- A dog playing in the park
- A group of people standing near a building
- Together with their pictures

---

*End of Instructions*
