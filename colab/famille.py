"""Famille Carter : Hunyuan3D-2.1 complet (forme + texture) sur le GPU T4 de Colab (16 Go).

Dans une cellule Colab (GPU T4) :
    !rm -rf /content/repo && git clone -q https://github.com/amineZEZ/la-veilleuse-rendu /content/repo
    %run /content/repo/colab/famille.py

Entrées : images détourées déposées dans /content/entrees (sinon une fenêtre d'envoi s'ouvre).
Sorties : un .zip par personnage, téléchargé dès qu'il est prêt, contenant
  - forme.glb (forme haute définition, sans texture),
  - modele/ (maillage allégé + textures couleur et métal/rugosité, format OBJ),
  - vues/ (les vues peintes par l'IA + angles de caméra), pour refaire la texture sur le Mac si besoin.

Pour tenir dans 16 Go au lieu des 21 Go annoncés : vues en 512 px, texture finale 2048 px, modèles chargés
sur la carte seulement quand ils servent, et agrandissement des vues fait simplement (Lanczos) au lieu de
Real-ESRGAN, qui ne s'installe plus proprement sur le Python de Colab.
"""
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import types
import zipfile

H = "/content/Hunyuan3D-2.1"
ENTREES = "/content/entrees"
SORTIES = "/content/sorties"


def sh(cmd, cwd=None):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


# ---------------------------------------------------------------- installation (une fois par machine)
if not os.path.exists(f"{H}/.installe"):
    if not os.path.exists(H):
        sh(f"git clone -q --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 {H}")
    sh("pip install -q transformers==4.46.3 diffusers==0.30.0 huggingface_hub==0.30.2 accelerate==1.1.1 "
       "omegaconf einops trimesh pymeshlab xatlas pygltflib opencv-python-headless scikit-image pybind11 ninja")
    sh("pip install -q --no-build-isolation .", cwd=f"{H}/hy3dpaint/custom_rasterizer")
    # le script fourni compte sur python3-config, absent de Colab : on donne nous-mêmes le bon suffixe
    sh("c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) mesh_inpaint_processor.cpp "
       "-o mesh_inpaint_processor$(python -c \"import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))\")",
       cwd=f"{H}/hy3dpaint/DifferentiableRenderer")
    open(f"{H}/.installe", "w").write("ok")

os.makedirs(ENTREES, exist_ok=True)
os.makedirs(SORTIES, exist_ok=True)
for f in glob.glob("/content/*_detoure.png"):      # images déposées par le panneau « Fichiers »
    shutil.move(f, os.path.join(ENTREES, os.path.basename(f)))
try:
    from google.colab import files as _colab
except ImportError:
    _colab = None
if not glob.glob(f"{ENTREES}/*.png") and _colab:
    for nom, data in _colab.upload().items():
        open(os.path.join(ENTREES, nom), "wb").write(data)
images = sorted(glob.glob(f"{ENTREES}/*.png"))
print("Personnages :", [os.path.basename(i) for i in images], flush=True)

sys.path.insert(0, f"{H}/hy3dshape")
sys.path.insert(0, f"{H}/hy3dpaint")
os.chdir(H)
import torch  # noqa: E402
from PIL import Image  # noqa: E402

print("GPU :", torch.cuda.get_device_name(0), flush=True)


def nom_de(f):
    return os.path.splitext(os.path.basename(f))[0].replace("_detoure", "")


# ---------------------------------------------------------------- 1. formes
_charger = torch.load


def _charger_mmap(*a, **k):       # le modèle (7,4 Go) déborderait les 12 Go de RAM s'il était copié
    k.setdefault("mmap", True)
    return _charger(*a, **k)


torch.load = _charger_mmap
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402
from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover  # noqa: E402

a_faire = [f for f in images if not os.path.exists(f"{SORTIES}/{nom_de(f)}/forme.glb")]
if a_faire:
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)
    forme = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1", device="cuda",
                                                            dtype=torch.float16)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    for f in a_faire:
        n = nom_de(f)
        os.makedirs(f"{SORTIES}/{n}", exist_ok=True)
        t = time.time()
        mesh = forme(image=Image.open(f).convert("RGBA"), num_inference_steps=50, guidance_scale=5.0,
                     octree_resolution=384, num_chunks=20000, generator=torch.manual_seed(1234),
                     output_type="trimesh")[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=400000)
        mesh.export(f"{SORTIES}/{n}/forme.glb")
        print(f"FORME {n} : {len(mesh.faces)} faces, {time.time() - t:.0f} s", flush=True)
        torch.cuda.empty_cache()
    del forme
torch.load = _charger
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2. textures
sys.modules.setdefault("bpy", types.ModuleType("bpy"))    # seul l'export .glb s'en sert ; on ne l'appelle pas
import utils.image_super_utils as _isu  # noqa: E402


class _Agrandir:
    def __init__(self, *a, **k):
        pass

    def __call__(self, im):
        return im.resize((im.width * 4, im.height * 4), Image.LANCZOS)


_isu.imageSuperNet = _Agrandir
import textureGenPipeline as tgp  # noqa: E402

tgp.imageSuperNet = _Agrandir
cfg = tgp.Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
cfg.texture_size = 2048
cfg.render_size = 2048
peintre = tgp.Hunyuan3DPaintPipeline(cfg)
mv = peintre.models["multiview_model"]
try:
    mv.pipeline.to("cpu")
    mv.pipeline.enable_model_cpu_offload()  # chaque réseau ne monte sur la carte que pendant son calcul
    print("Chargement à la demande activé", flush=True)
except Exception as e:
    print("Chargement à la demande impossible, tout reste sur la carte :", e, flush=True)
    mv.pipeline.to("cuda")
gc.collect()
torch.cuda.empty_cache()

# garde une copie des vues peintes (pour pouvoir refaire l'assemblage de la texture sur le Mac)
_vues = {}
_vp = peintre.view_processor
_bake = _vp.bake_from_multiview


def _bake_garde(imgs, elevs, azims, poids, *a, **k):
    _vues.setdefault("lots", []).append((imgs, list(elevs), list(azims), list(poids)))
    return _bake(imgs, elevs, azims, poids, *a, **k)


_vp.bake_from_multiview = _bake_garde

for f in images:
    n = nom_de(f)
    d = f"{SORTIES}/{n}"
    if os.path.exists(f"{d}.zip"):
        continue
    t = time.time()
    _vues.clear()
    os.makedirs(f"{d}/modele", exist_ok=True)
    shutil.copy(f"{d}/forme.glb", f"{d}/modele/forme.glb")
    try:
        peintre(mesh_path=f"{d}/modele/forme.glb", image_path=Image.open(f), output_mesh_path=f"{d}/modele/{n}.obj",
                use_remesh=True, save_glb=False)
        print(f"TEXTURE {n} : {time.time() - t:.0f} s", flush=True)
    except Exception as e:      # même en cas d'échec de l'assemblage, on garde les vues déjà peintes
        print(f"ÉCHEC TEXTURE {n} : {type(e).__name__}: {e}", flush=True)
    os.remove(f"{d}/modele/forme.glb")
    os.makedirs(f"{d}/vues", exist_ok=True)
    cams = []
    for k, (imgs, elevs, azims, poids) in enumerate(_vues.get("lots", [])):
        canal = ["albedo", "mr"][k] if k < 2 else f"lot{k}"
        for i, im in enumerate(imgs):
            im.save(f"{d}/vues/{canal}_{i}.png")
        cams.append({"canal": canal, "elevations": elevs, "azimuts": azims, "poids": poids})
    json.dump(cams, open(f"{d}/vues/cameras.json", "w"), indent=1)
    with zipfile.ZipFile(f"{d}.zip", "w") as z:
        for g in glob.glob(f"{d}/**/*", recursive=True):
            if os.path.isfile(g):
                z.write(g, os.path.relpath(g, SORTIES))
    if _colab:
        _colab.download(f"{d}.zip")
    print(f"PRÊT {n}.zip", flush=True)
    gc.collect()
    torch.cuda.empty_cache()

print("FINI", flush=True)
