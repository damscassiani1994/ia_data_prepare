# Deep Learning with PyTorch — Practice Notebooks

A collection of Jupyter notebooks covering the main Deep Learning topics using PyTorch, from fundamentals to advanced computer vision and NLP models.

---

## Topics Covered

### 1. Data Preparation and Fundamentals
**`main.ipynb`**
- Tabular data exploration and preprocessing with Pandas
- Categorical variable encoding with `OneHotEncoder` (sklearn)
- Dataset: Premier League statistics (`premier_league_data.csv`)

---

### 2. Introduction to PyTorch and Tensors
**`torch_imagen_clasification.ipynb`**
- Tensor creation and initialization
- Basic operations: `torch.tensor`, `torch.from_numpy`, `torch.ones_like`, `torch.rand`
- PyTorch API fundamentals

---

### 3. Image Classification with a Custom CNN
**`torch_my_image_clasification.ipynb`**
- Full image classification pipeline with torchvision
- Data transformations: resize, center crop, horizontal flip, rotation, color jitter
- Training a custom CNN with `nn.Module`
- Saved models in `models/`: Premier League logo classifier, animal classifier, CIFAR-10 network
- Custom dataset: Premier League team logos (Arsenal, Chelsea, Liverpool, ManCity, ManUtd, Aston Villa)

---

### 4. NLP — Bag of Words Classifier
**`nlp_bag_of_words_classifier.ipynb`**
- Spanish/English text classification with a BoW model
- `BoWClassifier` implementation with `nn.Linear` + `log_softmax`
- Manual vocabulary building and sentence vectorization

---

### 5. NLP — Word Embeddings
**`nlp_word_embeddings.ipynb`**
- Word embeddings with `nn.Embedding`
- N-gram language model (`NGramLanguageModeler`)
- Training with SGD and `NLLLoss`

---

### 6. NLP — Sequence Models with LSTM
**`nlp_sequence_models_tutorial.ipynb`**
- Introduction to LSTM with `nn.LSTM`
- Application to Part-of-Speech Tagging
- `LSTMTagger` model with embeddings + LSTM + linear layer

---

### 7. NLP — BiLSTM + Conditional Random Field (CRF)
**`lstm_conditional_random_field_discussion.ipynb`**
- BiLSTM-CRF model for Named Entity Recognition (NER)
- Manual implementation of the Viterbi algorithm and `log_sum_exp`
- BIO tagging scheme (Begin-Inside-Outside)

---

### 8. Computer Vision — Object Detection and Instance Segmentation with Mask R-CNN
**`mask_r_cnn_detention.ipynb`** / **`mask_r_cnn_detention.py`**
- Object detection and instance segmentation with Mask R-CNN
- Fine-tuning of `maskrcnn_resnet50_fpn` pretrained on COCO
- Dataset: PennFudan pedestrian dataset (images + masks)
- Reference utilities in `references/detection/`: engine, COCO eval, transforms

---

### 9. LLM Fine-tuning with LoRA (QLoRA) (PENDING)
**`train_llm_with_pythorch.ipynb`**
- Fine-tuning large language models with `SFTTrainer` (TRL)
- Efficient adaptation technique **LoRA** with `peft` (PEFT / QLoRA)
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Instruction dataset: `tatsu-lab/alpaca`
- `TrainingArguments` configuration: batch size, learning rate, warmup, logging

---

## Project Structure

```
data_prepare/
├── data/                          # Datasets (CIFAR-10, Premier League CSV)
├── data_test/premier_league_teams/ # Logo images for classification
├── models/                        # Trained models (.pth)
├── references/detection/          # Object detection utilities
├── main.ipynb                     # Tabular data preprocessing
├── torch_imagen_clasification.ipynb
├── torch_my_image_clasification.ipynb
├── nlp_bag_of_words_classifier.ipynb
├── nlp_word_embeddings.ipynb
├── nlp_sequence_models_tutorial.ipynb
├── lstm_conditional_random_field_discussion.ipynb
├── mask_r_cnn_detention.ipynb
└── train_llm_with_pythorch.ipynb
```

## Technologies

- **PyTorch** — main deep learning framework
- **torchvision** — models and transforms for computer vision
- **Transformers / TRL / PEFT** — LLM fine-tuning
- **scikit-learn / Pandas** — tabular data preprocessing
- **Matplotlib / PIL** — image visualization
