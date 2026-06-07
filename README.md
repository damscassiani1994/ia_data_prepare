# Deep Learning con PyTorch — Cuadernos de Práctica

Colección de notebooks de Jupyter que cubre los principales temas de Deep Learning usando PyTorch, desde fundamentos hasta modelos avanzados de visión por computadora y NLP.

---

## Temas cubiertos

### 1. Preparación de datos y fundamentos
**`main.ipynb`**
- Exploración y preprocesamiento de datos tabulares con Pandas
- Codificación de variables categóricas con `OneHotEncoder` (sklearn)
- Dataset: estadísticas de la Premier League (`premier_league_data.csv`)

---

### 2. Introducción a PyTorch y Tensores
**`torch_imagen_clasification.ipynb`**
- Creación e inicialización de tensores
- Operaciones básicas: `torch.tensor`, `torch.from_numpy`, `torch.ones_like`, `torch.rand`
- Fundamentos de la API de PyTorch

---

### 3. Clasificación de imágenes con CNN personalizada
**`torch_my_image_clasification.ipynb`**
- Pipeline completo de clasificación de imágenes con torchvision
- Transformaciones de datos: redimensionado, recorte, flip horizontal, rotación, color jitter
- Entrenamiento de una CNN propia con `nn.Module`
- Modelos guardados en `models/`: clasificador de logos de la Premier League, clasificador de animales, red CIFAR-10
- Dataset custom: logos de equipos Premier League (Arsenal, Chelsea, Liverpool, ManCity, ManUtd, Aston Villa)

---

### 4. NLP — Clasificador Bag of Words
**`nlp_bag_of_words_classifier.ipynb`**
- Clasificación de texto español/inglés con modelo BoW
- Implementación de `BoWClassifier` con `nn.Linear` + `log_softmax`
- Construcción manual del vocabulario y vectorización de frases

---

### 5. NLP — Word Embeddings
**`nlp_word_embeddings.ipynb`**
- Embeddings de palabras con `nn.Embedding`
- Modelo de lenguaje N-gramas (`NGramLanguageModeler`)
- Entrenamiento con SGD y `NLLLoss`

---

### 6. NLP — Modelos de secuencia con LSTM
**`nlp_sequence_models_tutorial.ipynb`**
- Introducción a LSTM con `nn.LSTM`
- Aplicación al etiquetado gramatical (Part-of-Speech Tagging)
- Modelo `LSTMTagger` con embeddings + LSTM + capa lineal

---

### 7. NLP — BiLSTM + Conditional Random Field (CRF)
**`lstm_conditional_random_field_discussion.ipynb`**
- Modelo BiLSTM-CRF para reconocimiento de entidades nombradas (NER)
- Implementación manual del algoritmo de Viterbi y `log_sum_exp`
- Etiquetado BIO (Begin-Inside-Outside)

---

### 8. Visión por computadora — Detección e instancias con Mask R-CNN
**`mask_r_cnn_detention.ipynb`** / **`mask_r_cnn_detention.py`**
- Detección de objetos e segmentación de instancias con Mask R-CNN
- Fine-tuning de `maskrcnn_resnet50_fpn` preentrenado en COCO
- Dataset: PennFudan pedestrian dataset (imágenes + máscaras)
- Utilities de referencia en `references/detection/`: engine, COCO eval, transforms

---

### 9. Fine-tuning de LLMs con LoRA (QLoRA)
**`train_llm_with_pythorch.ipynb`**
- Fine-tuning de modelos de lenguaje grandes con `SFTTrainer` (TRL)
- Técnica de adaptación eficiente **LoRA** con `peft` (PEFT / QLoRA)
- Modelo base: `Qwen/Qwen2.5-7B-Instruct`
- Dataset de instrucciones: `tatsu-lab/alpaca`
- Configuración de `TrainingArguments`: batch size, learning rate, warmup, logging

---

## Estructura del proyecto

```
data_prepare/
├── data/                          # Datasets (CIFAR-10, Premier League CSV)
├── data_test/premier_league_teams/ # Imágenes de logos para clasificación
├── models/                        # Modelos entrenados (.pth)
├── references/detection/          # Utilidades para detección de objetos
├── main.ipynb                     # Preprocesamiento de datos tabulares
├── torch_imagen_clasification.ipynb
├── torch_my_image_clasification.ipynb
├── nlp_bag_of_words_classifier.ipynb
├── nlp_word_embeddings.ipynb
├── nlp_sequence_models_tutorial.ipynb
├── lstm_conditional_random_field_discussion.ipynb
├── mask_r_cnn_detention.ipynb
└── train_llm_with_pythorch.ipynb
```

## Tecnologías

- **PyTorch** — framework principal de deep learning
- **torchvision** — modelos y transformaciones para visión
- **Transformers / TRL / PEFT** — fine-tuning de LLMs
- **scikit-learn / Pandas** — preprocesamiento de datos tabulares
- **Matplotlib / PIL** — visualización de imágenes
