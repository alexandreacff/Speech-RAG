"""Evaluation script for Speech RAG Retrieval"""

import argparse
import yaml
import torch
import json
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
from src.inference import SpeechRetriever


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_ground_truth(metadata_path: str, audio_dir: str) -> Dict[str, Dict]:
    """
    Carrega ground truth: mapeia query (question) -> audio_path correto
    
    Returns:
        Dict: {query_id: {"query": question, "correct_audio": audio_path, "id": id}}
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
                    ground_truth[query_id] = {
                        "query": question,
                        "correct_audio": normalized_path,
                        "id": query_id
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
    
    # Evaluate
    results = evaluate_retrieval(
        retriever=retriever,
        ground_truth=ground_truth,
        k_values=args.k,
        max_samples=args.max_samples
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for k in args.k:
        print(f"Recall@{k}:     {results[f'recall@{k}']:.4f} ({results[f'top_{k}_accuracy']*100:.2f}% correct)")
        print(f"Precision@{k}:  {results[f'precision@{k}']:.4f}")
    print(f"MRR:            {results['mrr']:.4f}")
    print(f"Total samples:  {results['num_samples']}")
    print("="*60)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

