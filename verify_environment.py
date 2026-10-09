"""
Preflight Environment & Model Verification Utility for MLUA Dental Caries AI
Validates local dependencies, environment variables, NLU models, and canonical
PyTorch checkpoint files (EXP-MLUA-003_E75_BEST.pth).
"""

import sys
import os
import hashlib
import shutil
from pathlib import Path

EXPECTED_E75_SHA256 = "cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb"
EXPECTED_E75_SIZE = 370840935

CANDIDATE_CHECKPOINT_PATHS = [
    Path("checkpoints/EXP-MLUA-003_E75_BEST.pth"),
    Path("outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth"),
    Path("FINAL_REVIEW_PACKAGE/09_TECHNICAL_EVIDENCE/checkpoint/EXP-MLUA-003_E75_BEST.pth")
]

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest().lower()

def check_checkpoint() -> bool:
    print("\n[1/4] Checking Canonical Active Model Checkpoint (EXP-MLUA-003_E75_BEST.pth)...")
    
    found_paths = [p for p in CANDIDATE_CHECKPOINT_PATHS if p.exists() and p.stat().st_size > 0]
    
    if not found_paths:
        print("  [!] STATUS: Checkpoint NOT found in default paths.")
        print("      Note: GitHub enforces a strict 100 MB per-file limit; because")
        print("      EXP-MLUA-003_E75_BEST.pth is ~370.8 MB, *.pth is excluded from git commits.")
        print("      If you have the review archive, extract FINAL_REVIEW_PACKAGE.zip")
        print("      or place EXP-MLUA-003_E75_BEST.pth into checkpoints/.")
        return False
        
    source_path = found_paths[0]
    file_size = source_path.stat().st_size
    print(f"  [+] Found checkpoint at: {source_path} ({file_size:,} bytes)")
    
    print("  [*] Verifying SHA-256 checksum...")
    actual_hash = compute_sha256(source_path)
    if actual_hash == EXPECTED_E75_SHA256:
        print(f"  [+] SHA-256 VERIFIED: {actual_hash[:16]}... (100% Match)")
    else:
        print(f"  [!] SHA-256 MISMATCH:")
        print(f"      Expected: {EXPECTED_E75_SHA256}")
        print(f"      Actual:   {actual_hash}")
        return False
        
    # Synchronize to standard locations if missing
    for target in CANDIDATE_CHECKPOINT_PATHS:
        if not target.exists() or target.stat().st_size == 0:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source_path, target)
                print(f"  [+] Synchronized checkpoint to: {target}")
            except Exception as e:
                print(f"  [-] Note: Could not copy to {target}: {e}")
                
    return True

def check_nlu_models() -> bool:
    print("\n[2/4] Checking Local Conversational NLU Models...")
    nlu_models = [
        Path("backend/nlp/models/conversational_v2_classifier.pkl"),
        Path("backend/nlp/models/local_intent_classifier.pkl")
    ]
    all_ok = True
    for model_path in nlu_models:
        if model_path.exists() and model_path.stat().st_size > 0:
            print(f"  [+] Found {model_path.name} ({model_path.stat().st_size:,} bytes)")
        else:
            print(f"  [!] Missing or empty: {model_path}")
            all_ok = False
    return all_ok

def check_env_configuration() -> bool:
    print("\n[3/4] Checking Environment Configuration...")
    env_file = Path(".env")
    if env_file.exists():
        print("  [+] .env file detected.")
    else:
        example = Path(".env.example")
        if example.exists():
            shutil.copy2(example, env_file)
            print("  [+] Created default .env from .env.example.")
        else:
            print("  [!] .env file missing.")
    return True

def check_packages() -> bool:
    print("\n[4/4] Checking Core Python Dependencies...")
    required = ["torch", "fastapi", "uvicorn", "sklearn", "pydantic", "dotenv", "pytest", "httpx"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  [+] {pkg}: OK")
        except ImportError:
            print(f"  [-] {pkg}: NOT INSTALLED")
            missing.append(pkg)
            
    if missing:
        print(f"\n  [!] Install missing packages with: pip install -r requirements.txt")
        return False
    return True

def main():
    print("=" * 65)
    print("  MLUA Dental Caries Clinical AI - Environment & Preflight Check")
    print("=" * 65)
    
    ckpt_ok = check_checkpoint()
    nlu_ok = check_nlu_models()
    env_ok = check_env_configuration()
    pkg_ok = check_packages()
    
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Model Checkpoint:   {'READY (Verified)' if ckpt_ok else 'ACTION REQUIRED'}")
    print(f"  NLU Models:         {'READY' if nlu_ok else 'MISSING'}")
    print(f"  Environment:        {'CONFIGURED' if env_ok else 'WARNING'}")
    print(f"  Python Packages:    {'INSTALLED' if pkg_ok else 'ACTION REQUIRED'}")
    
    print("\n  RUNNING THE PROJECT:")
    print("  1. Backend Server:  uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload")
    print("  2. Frontend UI:     cd frontend && npm run dev")
    print("  3. Verification:    pytest backend/test_chat_api.py -v")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
