# Speech-RAG

Treinamento de um retriever de fala para RAG (Retrieval-Augmented Generation), alinhando embeddings de áudio com embeddings de texto via distilação.

## Visão geral

Este projeto treina um pipeline com:
- `TextEncoder`: gera embedding da pergunta em texto.
- `SpeechEncoder` (HuBERT): gera representação da fala.
- `SpeechAdapter`: projeta a representação de fala para o mesmo espaço vetorial do texto.
- `DistillationLoss`: otimiza similaridade entre embeddings de texto e fala.

O objetivo é recuperar áudios relevantes para uma query textual e, opcionalmente, usar esses resultados em uma etapa de geração.

## Estrutura do repositório

```text
new-speech-rag/
├── config/
│   └── config.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── src/
│   ├── data/
│   └── models/
├── training/
│   ├── losses.py
│   └── trainer.py
└── test/
```

## Requisitos

- Python 3.10+
- CUDA (opcional, mas recomendado)
- `ffmpeg` instalado no sistema

Dependências Python principais:
- `torch`
- `torchaudio`
- `transformers`
- `sentence-transformers`
- `faiss-cpu` ou `faiss-gpu`
- `tqdm`
- `pyyaml`
- `numpy`
- `soundfile`
- `wandb`

Exemplo de instalação rápida:

```bash
pip install torch torchaudio transformers sentence-transformers faiss-cpu tqdm pyyaml numpy soundfile wandb
```

## Dataset esperado

O `scripts/train.py` espera os arquivos em `paths.data_dir` com esta estrutura:

```text
<DATA_DIR>/
├── spoken_train-v1.1.json
├── spoken_test-v1.1.json
├── train_wav/
└── dev_wav/
```

No estado atual do projeto, `config/config.yaml` aponta para:

```yaml
paths:
  data_dir: "/workspace/interspeech_2026/speech-rag/src/data"
```

Se necessário, ajuste para o seu ambiente.

## Configuração

Edite `config/config.yaml` para controlar:
- modelos (`models.text_encoder`, `models.speech_encoder`)
- treino (`training.batch_size`, `learning_rate`, `gradient_accumulation_steps`, etc.)
- geração (`generation.*`)
- caminhos (`paths.data_dir`, `paths.output_dir`)

## Treinamento

```bash
python scripts/train.py --config config/config.yaml
```

Opções úteis:

```bash
# Desabilita wandb
python scripts/train.py --config config/config.yaml --no-wandb

# Força device
python scripts/train.py --config config/config.yaml --device cuda

# Retoma de checkpoint
python scripts/train.py --config config/config.yaml --resume /caminho/para/checkpoint.pt
```

## Avaliação

```bash
python scripts/evaluate.py \
  --config config/config.yaml \
  --checkpoint outputs/final_checkpoint.pt \
  --audio-dir /caminho/para/dev_wav \
  --k 1 5 10 \
  --output evaluation_results.json
```

## Inferência

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint outputs/final_checkpoint.pt \
  --audio-dir /caminho/para/dev_wav \
  --query "What is the main symptom?" \
  --k 10
```

Para gerar resposta condicionada por áudio recuperado:

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint outputs/final_checkpoint.pt \
  --audio-dir /caminho/para/dev_wav \
  --query "What is the main symptom?" \
  --k 10 \
  --generate
```

## Monitoramento com Weights & Biases

Por padrão, o treinamento loga no W&B (projeto `speech-rag`).

Para desativar:

```bash
python scripts/train.py --config config/config.yaml --no-wandb
```

## Troubleshooting

### CUDA Out Of Memory (OOM)

Se ocorrer OOM (como no HuBERT large), tente nesta ordem:

1. Reduzir `training.batch_size` no `config/config.yaml`.
2. Aumentar `training.gradient_accumulation_steps` para manter batch efetivo.
3. Habilitar mixed precision (`training.use_amp: true`).
4. Congelar encoder de fala (`training.finetune_speech_encoder: false`) para reduzir memória.
5. Exportar:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

6. Fechar processos que ocupam GPU (`nvidia-smi`).

### Caminhos de dados inválidos

Verifique se `paths.data_dir` está correto e contém os arquivos JSON e pastas de áudio esperados.

## Status

- Treinamento: funcional
- Avaliação: funcional com checkpoint e dados válidos
- Inferência/Geração: requer módulos de inferência presentes em `src/inference`

## Licença

Defina aqui a licença do projeto (por exemplo, MIT, Apache-2.0, etc.).
