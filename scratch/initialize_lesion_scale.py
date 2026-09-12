import sys
from pathlib import Path

base_dir = Path("c:/Users/devin/MLUA")
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.mlua.evaluation.lesion_scale_analysis import LesionScaleAnalyzer

out_dir = base_dir / "outputs" / "evaluation" / "lesion_scale"
analyzer = LesionScaleAnalyzer()
analyzer.export_schemas(out_dir)
print(f"Initialized lesion scale schemas in {out_dir}")
