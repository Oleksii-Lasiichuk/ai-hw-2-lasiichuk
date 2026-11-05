import json
import sys
from pathlib import Path

# usage: python scripts/build_kaggle.py 02_ultralytics [RUN] -> runs/kaggle/02_ultralytics[_RUN].ipynb with src/ inlined
root = Path(__file__).resolve().parents[1]
name, run = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
notebook = json.loads((root / f"notebooks/{name}.ipynb").read_text())
install = ("# --no-deps keeps Kaggle's own torch, numpy and Pillow untouched\n"
           "!pip install -q --no-deps ultralytics==8.3.221 ultralytics-thop==2.2.4\n"
           "!pip install -q faster-coco-eval==1.7.0 polars==2.1.0\n"
           "!mkdir -p src && touch src/__init__.py")
sources = [install] + [f"%%writefile src/{p.name}\n{p.read_text()}" for p in sorted((root / "src").glob("[!_]*.py"))]
setup = [{"cell_type": "code", "id": f"setup-{i}", "metadata": {}, "execution_count": None, "outputs": [],
          "source": s.splitlines(keepends=True)} for i, s in enumerate(sources)]
notebook["cells"] = notebook["cells"][:1] + setup + notebook["cells"][1:]
if run:
    for cell in notebook["cells"]:
        cell["source"] = [f'RUN = "{run}"\n' if line.startswith("RUN = ") else line for line in cell["source"]]
out = root / f"runs/kaggle/{name}{'_' + run if run else ''}.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1) + "\n")
print(out)
