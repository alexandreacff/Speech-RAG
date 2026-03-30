"""Inference script for Speech Retriever"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
from src.inference import SpeechRetriever, SpeechRAGPipeline, AudioConditionedGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Speech Retriever Inference")
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
        "--query",
        type=str,
        default=None,
        help="Text query to search"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to return"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Enable response generation using Qwen-Audio-Chat (requires retrieved audios)"
    )
    parser.add_argument(
        "--top-k-audio",
        type=int,
        default=None,
        help="Number of top audio passages to use for generation (uses config if not specified)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (uses config if not specified)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (uses config if not specified)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
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
    
    # Build or load index (sempre tenta reutilizar primeiro)
    audio_dir_to_use = args.audio_dir
    
    # Gerar caminho padrão do índice se não fornecido
    if not args.index and audio_dir_to_use:
        # Usa o nome do diretório de áudio para criar nome do índice
        audio_dir_name = Path(audio_dir_to_use).name
        default_index_path = f"indices/{audio_dir_name}_index.faiss"
        index_path = Path(default_index_path)
    elif args.index:
        index_path = Path(args.index)
    else:
        print("Error: Either --index or --audio-dir must be provided")
        return
    
    # Sempre tenta carregar índice existente primeiro
    if index_path.exists():
        print(f"✓ Loading existing index from {index_path}...")
        retriever.load_index(str(index_path))
        print(f"  Index loaded with {retriever.index.ntotal} vectors")
    elif audio_dir_to_use and Path(audio_dir_to_use).exists():
        # Só constrói se não existir
        print(f"Index not found at {index_path}. Building new index...")
        print(f"Building index from audio files in {audio_dir_to_use}...")
        audio_dir = Path(audio_dir_to_use)
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.flac"))
        
        if not audio_files:
            print(f"No audio files found in {audio_dir_to_use}")
            return
        
        print(f"Found {len(audio_files)} audio files")
        retriever.build_index(audio_files)
        
        # Salva automaticamente para reutilização futura
        index_path.parent.mkdir(parents=True, exist_ok=True)
        retriever.save_index(str(index_path))
        print(f"✓ Index saved to {index_path} for future reuse")
    else:
        print(f"Error: Could not find index at {index_path} and audio_dir not provided or invalid")
        return
    
    # Search if query provided
    if args.query:
        print(f"\nSearching for: '{args.query}'")
        results = retriever.search(args.query, k=args.k)
        
        print(f"\nTop {len(results)} results:")
        print("-" * 80)
        for result in results:
            print(f"Rank {result['rank']}: {result['audio_path']}")
            print(f"  Score: {result['score']:.4f}")
            if result.get('metadata'):
                print(f"  Metadata: {result['metadata']}")
            print()
        
        # Generate response if --generate is enabled
        if args.generate:
            print("\n" + "=" * 80)
            print("Generating response using Qwen-Audio-Chat...")
            print("=" * 80)
            
            # Load generation config
            gen_config = config.get("generation", {})
            top_k_audio = args.top_k_audio or gen_config.get("top_k_audio", 3)
            temperature = args.temperature if args.temperature is not None else gen_config.get("temperature", 0.7)
            max_new_tokens = args.max_new_tokens or gen_config.get("max_new_tokens", 512)
            generator_device = gen_config.get("device") or device
            
            # Create generator
            print("Loading Qwen-Audio-Chat generator...")
            generator = AudioConditionedGenerator(
                model_name=gen_config.get("model_name", "Qwen/Qwen-Audio-Chat"),
                device=generator_device
            )
            
            # Create RAG pipeline
            pipeline = SpeechRAGPipeline(
                retriever=retriever,
                generator=generator,
                top_k_audio=top_k_audio
            )
            
            # Generate response
            print(f"Using top {top_k_audio} audio passages for generation...")
            rag_result = pipeline.retrieve_and_generate(
                query=args.query,
                k=top_k_audio,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                return_retrieval_results=True
            )
            
            # Display generated response
            print("\n" + "-" * 80)
            print("Generated Response:")
            print("-" * 80)
            print(rag_result["response"])
            print("-" * 80)
            print(f"\nUsed {rag_result['num_audios']} audio passage(s) for generation")
            
            # Save results if output specified
            if args.output:
                output_data = {
                    "query": args.query,
                    "k": args.k,
                    "retrieval_results": results,
                    "generation": {
                        "response": rag_result["response"],
                        "audio_paths": rag_result["audio_paths"],
                        "num_audios": rag_result["num_audios"],
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens
                    }
                }
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to {args.output}")
        else:
            # Save results if output specified (retrieval only)
            if args.output:
                output_data = {
                    "query": args.query,
                    "k": args.k,
                    "results": results
                }
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to {args.output}")
    else:
        print("No query provided. Use --query to search.")
        if args.generate:
            print("Note: --generate requires --query to be provided.")
        print("Index is ready for queries.")


if __name__ == "__main__":
    main()

