"""Génère les modèles 3D du film avec Hunyuan3D-2.1 (forme seule, haute résolution) sur le GPU de Colab.

Dans une cellule Colab (GPU T4) :
    !rm -rf /content/repo && git clone -q https://github.com/amineZEZ/la-veilleuse-rendu /content/repo
    %run /content/repo/colab/generer.py
Chaque .glb est téléchargé dès qu'il est prêt (une machine Colab peut être recyclée à tout moment).

Entrées : colab/entrees/*.png (images détourées). Sortie : /content/modeles.zip (un .glb par image).
"""
import glob
import os
import subprocess
import sys
import time
import zipfile

H = "/content/Hunyuan3D-2.1"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


if not os.path.exists(H):
    sh(f"git clone -q --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 {H}")
    sh("pip install -q diffusers transformers accelerate trimesh pymeshlab omegaconf einops "
       "scikit-image pygltflib opencv-python-headless")

# Hunyuan3D-2.1 attend l'ancien nommage des poids de DINOv2 (transformers 4.46)
import importlib.metadata as _md  # noqa: E402
if not _md.version("transformers").startswith("4.46") or not _md.version("diffusers").startswith("0.31"):
    sh("pip install -q transformers==4.46.3 diffusers==0.31.0 huggingface_hub==0.26.5")

sys.path.insert(0, os.path.join(H, "hy3dshape"))
os.chdir(H)
import torch  # noqa: E402

# Le modèle (7,4 Go) ferait déborder les 12 Go de RAM de Colab s'il était chargé en entier puis copié :
# on le lit directement depuis le disque (mmap).
_charger = torch.load


def _charger_mmap(*a, **k):
    k.setdefault("mmap", True)
    return _charger(*a, **k)


torch.load = _charger_mmap
from PIL import Image  # noqa: E402
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402
from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover  # noqa: E402

try:   # lancé avec %run dans le carnet : chaque modèle part sur l'ordinateur dès qu'il est prêt
    from google.colab import files as _colab_files
except ImportError:
    _colab_files = None

print("GPU :", torch.cuda.get_device_name(0), flush=True)
# construit les réseaux directement sur la carte graphique en demi-précision : en pleine précision
# sur le processeur, le modèle (3,3 milliards de paramètres) prendrait ~13 Go de RAM
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1", device="cuda", dtype=torch.float16)
torch.set_default_dtype(torch.float32)
torch.set_default_device("cpu")
import gc  # noqa: E402
gc.collect()
out = "/content/sorties"
os.makedirs(out, exist_ok=True)
octree = int(os.environ.get("OCTREE", "384"))
for f in sorted(glob.glob("/content/repo/colab/entrees/*.png")):
    nom = os.path.splitext(os.path.basename(f))[0].replace("_detoure", "")
    t = time.time()
    img = Image.open(f).convert("RGBA")
    mesh = pipe(image=img, num_inference_steps=50, guidance_scale=5.0, octree_resolution=octree,
                num_chunks=20000, generator=torch.manual_seed(1234), output_type="trimesh")[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=400000)
    chemin = os.path.join(out, f"{nom}.glb")
    mesh.export(chemin)
    if _colab_files:
        _colab_files.download(chemin)
    print(f"OK {nom} : {len(mesh.faces)} faces, {time.time() - t:.0f} s", flush=True)
    torch.cuda.empty_cache()

with zipfile.ZipFile("/content/modeles.zip", "w") as z:
    for g in glob.glob(f"{out}/*.glb"):
        z.write(g, os.path.basename(g))
print("FINI /content/modeles.zip", flush=True)
