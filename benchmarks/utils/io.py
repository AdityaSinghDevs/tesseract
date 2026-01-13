import json 
from pathlib import Path
from typing import Dict, Any

def load_prompt(prompt_path : Path)-> str:
   """
    Load a benchmark prompt from a text file.

    Args:
        prompt_path: Path to the .txt prompt file

    Returns:
        Prompt string with surrounding whitespace removed
    """
   if not prompt_path.exists():
      raise FileNotFoundError(f"Prompt file nnot found: {prompt_path}")
   
   with open(prompt_path, "r", encoding='utf-8') as f:
      prompt = f.read()

   return prompt.strip

def write_raw_results(output_path: Path, data: Dict[str, Any])->None:
   """
    Write raw benchmark results to disk as JSON.

    Args:
        output_path: Path where the JSON file will be written
        data: Dictionary containing raw benchmark results and metadata
    """
   
   output_path.parent.mkdir(parents=True, exist_ok=True)

   with open(output_path, "w", encoding='utf-8') as f:
      json.dump(data,f, indent = 2)