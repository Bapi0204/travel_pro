import os
from dotenv import load_dotenv
load_dotenv()

import argparse
from travel_pro.agent import run_episode

def main():
    parser = argparse.ArgumentParser(description="Manual Evaluation Client for Travel Pro Agent")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1, help="Entropy Level (1: Happy, 2: Adversarial, 3: Chaos)")
    parser.add_argument("--key", type=str, help="OpenAI API Key (optional, will use env var if not provided)")
    
    args = parser.parse_args()
    
    api_key = args.key or os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        print(f">>> API Key set. Using GPT-4o for Level {args.level} evaluation.")
    else:
        print(f">>> No API Key found. Using heuristic fallback for Level {args.level} evaluation.")

    print("\n" + "="*50)
    print(f"RUNNING EVALUATION: LEVEL {args.level}")
    print("="*50)
    
    success = run_episode(args.level)
    
    print("\n" + "="*50)
    if success:
        print(f"RESULT: SUCCESS (Goal achieved, constraints respected)")
    else:
        print(f"RESULT: FAILURE (Goal failed or constraints violated)")
    print("="*50)

if __name__ == "__main__":
    main()
