"""Evaluation script for Speech RAG Retrieval"""

import argparse
import yaml
import torch
import json
import os
import re
import string
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
from src.inference import SpeechRetriever, SpeechRAGPipeline, AudioConditionedGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_ground_truth(metadata_path: str, audio_dir: str) -> Dict[str, Dict]:
    """
    Carrega ground truth: mapeia query (question) -> audio_path correto
    
    Returns:
        Dict: {query_id: {"query": question, "correct_audio": audio_path, "answers": [...]}}
    """
    print(f"Loading ground truth from {metadata_path}...")
    print(f"  Audio directory: {audio_dir}")
    with open(metadata_path, 'r') as f:
        raw_data = json.load(f)
    
    ground_truth = {}
    audio_dir_path = Path(audio_dir)
    total_qas = 0
    missing_files = 0
    
    for article_idx, article in enumerate(raw_data['data']):
        for para_idx, paragraph in enumerate(article['paragraphs']):
            for qa_idx, qa in enumerate(paragraph['qas']):
                total_qas += 1
                question = qa['question']
                query_id = f"{article_idx}_{para_idx}_{qa_idx}"
                filename = f"{query_id}.wav"
                audio_path = audio_dir_path / filename
                
                if audio_path.exists():
                    # Normalizar path para comparação consistente
                    normalized_path = str(Path(audio_path).resolve())

                    answers = []
                    for answer_item in qa.get("answers", []):
                        if isinstance(answer_item, dict):
                            answer_text = str(answer_item.get("text", "")).strip()
                            if answer_text:
                                answers.append(answer_text)

                    if not answers:
                        # Fallback para datasets em que answer pode vir em outro campo.
                        fallback_answer = str(qa.get("answer", "")).strip()
                        if fallback_answer:
                            answers.append(fallback_answer)

                    ground_truth[query_id] = {
                        "query": question,
                        "correct_audio": normalized_path,
                        "id": query_id,
                        "answers": answers,
                    }
                else:
                    missing_files += 1
                    if missing_files <= 5:  # Mostrar apenas os primeiros 5
                        print(f"  DEBUG: Missing audio file: {audio_path}")
    
    print(f"  Total QAs in metadata: {total_qas}")
    print(f"  Found audio files: {len(ground_truth)}")
    print(f"  Missing audio files: {missing_files}")
    
    # Mostrar alguns exemplos
    if ground_truth:
        print(f"\n  DEBUG: Sample ground truth entries (first 3):")
        for i, (qid, gt) in enumerate(list(ground_truth.items())[:3]):
            print(f"    [{i+1}] Query ID: {qid}")
            print(f"        Query: {gt['query'][:80]}...")
            print(f"        Correct audio: {gt['correct_audio']}")
    
    return ground_truth


def calculate_recall_at_k(retrieved_paths: List[str], correct_path: str, k: int) -> float:
    """Calcula Recall@K com normalização de paths"""
    top_k = retrieved_paths[:k]
    # Normalizar paths para comparação consistente
    correct_path_normalized = str(Path(correct_path).resolve())
    top_k_normalized = [str(Path(p).resolve()) for p in top_k]
    return 1.0 if correct_path_normalized in top_k_normalized else 0.0


def calculate_precision_at_k(retrieved_paths: List[str], correct_path: str, k: int) -> float:
    """Calcula Precision@K (assumindo apenas 1 relevante) com normalização"""
    top_k = retrieved_paths[:k]
    # Normalizar paths para comparação consistente
    correct_path_normalized = str(Path(correct_path).resolve())
    top_k_normalized = [str(Path(p).resolve()) for p in top_k]
    return 1.0 if correct_path_normalized in top_k_normalized else 0.0


def calculate_mrr(retrieved_paths: List[str], correct_path: str) -> float:
    """Calcula Mean Reciprocal Rank com normalização de paths"""
    # Normalizar paths para comparação consistente
    correct_path_normalized = str(Path(correct_path).resolve())
    retrieved_normalized = [str(Path(p).resolve()) for p in retrieved_paths]
    try:
        rank = retrieved_normalized.index(correct_path_normalized) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def normalize_answer(text: str) -> str:
    """Normalize text for QA metrics (similar to SQuAD)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match_score(prediction: str, references: List[str]) -> float:
    if not references:
        return 0.0
    pred_norm = normalize_answer(prediction)
    return 1.0 if any(pred_norm == normalize_answer(ref) for ref in references) else 0.0


def token_f1_score(prediction: str, references: List[str]) -> float:
    if not references:
        return 0.0

    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        if not ref_tokens:
            continue

        common = {}
        for token in pred_tokens:
            common[token] = min(pred_tokens.count(token), ref_tokens.count(token))
        overlap = sum(common.values())

        if overlap == 0:
            f1 = 0.0
        else:
            precision = overlap / max(1, len(pred_tokens))
            recall = overlap / max(1, len(ref_tokens))
            f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1

    return best_f1


def evaluate_generation(
    pipeline: SpeechRAGPipeline,
    ground_truth: Dict[str, Dict],
    top_k_audio: int,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    do_sample: bool,
    max_samples: Optional[int] = None,
) -> Dict:
    """Evaluate end-to-end generation with QA metrics (Exact Match and Token F1)."""
    samples_to_eval = list(ground_truth.items())
    if max_samples:
        samples_to_eval = samples_to_eval[:max_samples]

    em_scores: List[float] = []
    f1_scores: List[float] = []
    predictions: List[Dict] = []

    for query_id, gt in tqdm(samples_to_eval, desc="Generating"):
        query = gt["query"]
        answers = gt.get("answers", [])

        rag_result = pipeline.retrieve_and_generate(
            query=query,
            k=top_k_audio,
            return_retrieval_results=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            do_sample=do_sample,
        )

        prediction = rag_result["response"]
        em = exact_match_score(prediction, answers)
        f1 = token_f1_score(prediction, answers)

        em_scores.append(em)
        f1_scores.append(f1)

        predictions.append(
            {
                "query_id": query_id,
                "query": query,
                "prediction": prediction,
                "references": answers,
                "em": em,
                "token_f1": f1,
                "audio_paths": rag_result.get("audio_paths", []),
            }
        )

    count = len(samples_to_eval)
    return {
        "num_samples": count,
        "exact_match": sum(em_scores) / max(1, count),
        "token_f1": sum(f1_scores) / max(1, count),
        "predictions": predictions,
    }


def evaluate_retrieval(
    retriever: SpeechRetriever,
    ground_truth: Dict[str, Dict],
    k_values: List[int] = [1, 5, 10],
    max_samples: int = None
) -> Dict:
    """
    Avalia o retrieval usando ground truth.
    
    Args:
        retriever: SpeechRetriever instanciado
        ground_truth: Dict com queries e áudios corretos
        k_values: Lista de valores de K para calcular métricas
        max_samples: Número máximo de amostras para avaliar (None = todas)
    
    Returns:
        Dict com métricas agregadas
    """
    print(f"\nEvaluating on {len(ground_truth)} queries...")
    
    # DEBUG: Verificar paths no índice
    print(f"\n  DEBUG: Index contains {retriever.index.ntotal} vectors")
    if retriever.audio_paths:
        print(f"  DEBUG: Sample paths in index (first 3):")
        for i, path in enumerate(retriever.audio_paths[:3]):
            normalized = str(Path(path).resolve())
            print(f"    [{i+1}] {normalized}")
        
        # Verificar se paths do índice estão normalizados
        index_paths_normalized = [str(Path(p).resolve()) for p in retriever.audio_paths]
        gt_paths_normalized = [gt["correct_audio"] for gt in ground_truth.values()]
        
        # Verificar overlap
        index_paths_set = set(index_paths_normalized)
        gt_paths_set = set(gt_paths_normalized)
        overlap = index_paths_set & gt_paths_set
        
        print(f"  DEBUG: Path overlap analysis:")
        print(f"    Index paths: {len(index_paths_set)}")
        print(f"    Ground truth paths: {len(gt_paths_set)}")
        print(f"    Overlapping paths: {len(overlap)}")
        if len(overlap) < len(gt_paths_set):
            missing_in_index = gt_paths_set - index_paths_set
            print(f"    WARNING: {len(missing_in_index)} ground truth paths not in index!")
            if len(missing_in_index) <= 5:
                print(f"    Missing examples:")
                for path in list(missing_in_index)[:5]:
                    print(f"      - {path}")
    
    # Métricas acumuladas
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = []
        metrics[f"precision@{k}"] = []
    
    metrics["mrr"] = []
    
    # Processar cada query
    samples_to_eval = list(ground_truth.items())
    if max_samples:
        samples_to_eval = samples_to_eval[:max_samples]
        print(f"  DEBUG: Limiting evaluation to {max_samples} samples")
    
    correct_count = {f"top_{k}": 0 for k in k_values}
    
    # DEBUG: Contadores para análise
    debug_samples_shown = 0
    max_debug_samples = 5
    
    for query_id, gt in tqdm(samples_to_eval, desc="Evaluating"):
        query = gt["query"]
        correct_audio = gt["correct_audio"]
        
        # Buscar com RAG
        max_k = max(k_values)
        results = retriever.search(query, k=max_k)
        retrieved_paths = [r["audio_path"] for r in results]
        
        # DEBUG: Mostrar alguns exemplos detalhados
        if debug_samples_shown < max_debug_samples:
            print(f"\n  DEBUG: Sample evaluation [{debug_samples_shown + 1}]:")
            print(f"    Query ID: {query_id}")
            print(f"    Query: {query[:100]}...")
            print(f"    Correct audio: {Path(correct_audio).name}")
            print(f"    Top-{min(3, len(results))} retrieved:")
            for i, r in enumerate(results[:3]):
                retrieved_name = Path(r["audio_path"]).name
                score = r.get("score", "N/A")
                is_match = str(Path(r["audio_path"]).resolve()) == correct_audio
                match_str = "✓ MATCH" if is_match else "✗"
                print(f"      [{i+1}] {retrieved_name} (score: {score:.4f}) {match_str}")
            debug_samples_shown += 1
        
        # Calcular métricas
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_paths, correct_audio, k)
            precision = calculate_precision_at_k(retrieved_paths, correct_audio, k)
            metrics[f"recall@{k}"].append(recall)
            metrics[f"precision@{k}"].append(precision)
            
            if recall > 0:
                correct_count[f"top_{k}"] += 1
        
        mrr = calculate_mrr(retrieved_paths, correct_audio)
        metrics["mrr"].append(mrr)
    
    # DEBUG: Estatísticas intermediárias
    print(f"\n  DEBUG: Intermediate statistics:")
    for k in k_values:
        correct = correct_count[f"top_{k}"]
        total = len(samples_to_eval)
        print(f"    Top-{k} correct: {correct}/{total} ({100*correct/total:.2f}%)")
    
    # Calcular médias
    results = {}
    for k in k_values:
        results[f"recall@{k}"] = sum(metrics[f"recall@{k}"]) / len(metrics[f"recall@{k}"])
        results[f"precision@{k}"] = sum(metrics[f"precision@{k}"]) / len(metrics[f"precision@{k}"])
        results[f"top_{k}_accuracy"] = correct_count[f"top_{k}"] / len(samples_to_eval)
    
    results["mrr"] = sum(metrics["mrr"]) / len(metrics["mrr"])
    results["num_samples"] = len(samples_to_eval)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speech RAG Retrieval")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to adapter checkpoint"
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to saved FAISS index (if exists)"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory with audio files to index"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata JSON file (default: data_dir/spoken_test-v1.1.json)"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="K values for Recall@K and Precision@K"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--eval-generation",
        action="store_true",
        help="Evaluate end-to-end generation with Exact Match and Token F1"
    )
    parser.add_argument(
        "--generation-max-samples",
        type=int,
        default=None,
        help="Max samples for generation evaluation (None = use same as retrieval)"
    )
    parser.add_argument(
        "--top-k-audio",
        type=int,
        default=None,
        help="Top-K retrieved audios used for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max new tokens for generation"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p for generation sampling"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    data_root = config["paths"].get("data_dir", "src/data")
    
    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Paths
    metadata_path = args.metadata or os.path.join(data_root, "spoken_test-v1.1.json")
    audio_dir = args.audio_dir or os.path.join(data_root, "dev_wav")
    
    # Load models
    print("Loading models...")
    text_encoder = TextEncoder(
        model_name=config["models"]["text_encoder"],
        freeze=True
    )
    speech_encoder = SpeechEncoder(
        model_name=config["models"]["speech_encoder"],
        freeze=True
    )
    
    # Get embedding dimensions
    text_embedding_dim = text_encoder.embedding_dim
    speech_hidden_dim = speech_encoder.hidden_size
    
    # Create adapter
    adapter = SpeechAdapter(
        input_dim=speech_hidden_dim,
        output_dim=text_embedding_dim,
        downsample_factor=4
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create retriever
    retriever = SpeechRetriever(
        text_encoder=text_encoder,
        speech_encoder=speech_encoder,
        adapter=adapter,
        device=device
    )
    
    # Load or build index (sempre tenta reutilizar primeiro)
    audio_dir_to_use = args.audio_dir or audio_dir
    
    # Gerar caminho padrão do índice se não fornecido
    if not args.index:
        # Usa o nome do diretório de áudio para criar nome do índice
        audio_dir_name = Path(audio_dir_to_use).name
        default_index_path = f"indices/{audio_dir_name}_index.faiss"
        index_path = Path(default_index_path)
    else:
        index_path = Path(args.index)
    
    # Sempre tenta carregar índice existente primeiro
    if index_path.exists():
        print(f"✓ Loading existing index from {index_path}...")
        retriever.load_index(str(index_path))
        print(f"  Index loaded with {retriever.index.ntotal} vectors")
        print(f"  DEBUG: Index audio paths sample (first 3):")
        if retriever.audio_paths:
            for i, path in enumerate(retriever.audio_paths[:3]):
                print(f"    [{i+1}] {path}")
    elif audio_dir_to_use and Path(audio_dir_to_use).exists():
        # Só constrói se não existir
        print(f"Index not found at {index_path}. Building new index...")
        print(f"Building index from {audio_dir_to_use}...")
        audio_files = list(Path(audio_dir_to_use).glob("*.wav"))
        
        if not audio_files:
            print(f"Error: No audio files found in {audio_dir_to_use}")
            return
        
        print(f"Found {len(audio_files)} audio files")
        print(f"  DEBUG: Sample audio files (first 3):")
        for i, af in enumerate(audio_files[:3]):
            print(f"    [{i+1}] {af}")
        
        retriever.build_index(audio_files)
        
        # Salva automaticamente para reutilização futura
        index_path.parent.mkdir(parents=True, exist_ok=True)
        retriever.save_index(str(index_path))
        print(f"✓ Index saved to {index_path} for future reuse")
    else:
        print(f"Error: Could not find index at {index_path} and audio_dir not provided or invalid")
        return
    
    # Load ground truth
    ground_truth = load_ground_truth(metadata_path, audio_dir)
    print(f"Loaded {len(ground_truth)} ground truth samples")
    
    if len(ground_truth) == 0:
        print("Error: No ground truth samples found. Check metadata and audio paths.")
        return
    
    # Evaluate retrieval
    retrieval_results = evaluate_retrieval(
        retriever=retriever,
        ground_truth=ground_truth,
        k_values=args.k,
        max_samples=args.max_samples
    )
    
    # Print results
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*60)
    for k in args.k:
        print(f"Recall@{k}:     {retrieval_results[f'recall@{k}']:.4f} ({retrieval_results[f'top_{k}_accuracy']*100:.2f}% correct)")
        print(f"Precision@{k}:  {retrieval_results[f'precision@{k}']:.4f}")
    print(f"MRR:            {retrieval_results['mrr']:.4f}")
    print(f"Total samples:  {retrieval_results['num_samples']}")
    print("="*60)

    final_results = {
        "retrieval": retrieval_results,
    }

    # Optional generation evaluation
    if args.eval_generation:
        print("\nInitializing generator for end-to-end evaluation...")
        gen_config = config.get("generation", {})
        top_k_audio = args.top_k_audio or int(gen_config.get("top_k_audio", 3))
        temperature = args.temperature if args.temperature is not None else float(gen_config.get("temperature", 0.7))
        max_new_tokens = args.max_new_tokens or int(gen_config.get("max_new_tokens", 256))
        top_p = args.top_p if args.top_p is not None else float(gen_config.get("top_p", 0.9))
        do_sample = args.do_sample or bool(gen_config.get("do_sample", False))
        generation_max_samples = args.generation_max_samples

        generator = AudioConditionedGenerator(
            model_name=gen_config.get("model_name", config["models"].get("generator", "Qwen/Qwen-Audio-Chat")),
            device=gen_config.get("device") or device,
        )
        pipeline = SpeechRAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k_audio=top_k_audio,
        )

        generation_results = evaluate_generation(
            pipeline=pipeline,
            ground_truth=ground_truth,
            top_k_audio=top_k_audio,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            do_sample=do_sample,
            max_samples=generation_max_samples,
        )

        print("\n" + "="*60)
        print("GENERATION EVALUATION RESULTS")
        print("="*60)
        print(f"Exact Match:    {generation_results['exact_match']:.4f}")
        print(f"Token F1:       {generation_results['token_f1']:.4f}")
        print(f"Total samples:  {generation_results['num_samples']}")
        print("="*60)

        final_results["generation"] = {
            "exact_match": generation_results["exact_match"],
            "token_f1": generation_results["token_f1"],
            "num_samples": generation_results["num_samples"],
        }
        final_results["generation_predictions"] = generation_results["predictions"]
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

