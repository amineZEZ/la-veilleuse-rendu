"""Calcule des images du film sur la carte graphique de Colab (Cycles + OptiX).

Dans une cellule Colab (GPU T4) :
    !rm -rf /content/repo && git clone -q https://github.com/amineZEZ/la-veilleuse-rendu /content/repo
    IMAGES = "100"            # ou "1-120", ou "100,560,680"
    %run /content/repo/colab/rendu_gpu.py
Les images sont rangées dans /content/images puis envoyées par paquets (zip) sur l'ordinateur.
"""
import glob
import os
import subprocess
import time
import zipfile

B = "/content/blender/blender"
F = "/content/la_veilleuse.blend"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


if not os.path.exists(B):
    sh("curl -fsSL https://download.blender.org/release/Blender5.2/blender-5.2.2-linux-x64.tar.xz -o /content/b.tar.xz")
    sh("mkdir -p /content/blender && tar -xf /content/b.tar.xz -C /content/blender --strip-components=1 && rm /content/b.tar.xz")
    sh("apt-get -qq install -y libxrender1 libxi6 libxkbcommon0 libsm6 libgl1 libxxf86vm1 libxfixes3 libegl1 > /dev/null")
if not os.path.exists(F):
    sh(f"curl -fsSL https://github.com/amineZEZ/la-veilleuse-rendu/releases/download/v1/la_veilleuse.blend -o {F}")

spec = globals().get("IMAGES", os.environ.get("IMAGES", "100"))
images = []
for part in spec.split(","):
    if "-" in part:
        a, b = map(int, part.split("-"))
        images += list(range(a, b + 1))
    else:
        images.append(int(part))

gpu = ("import bpy; p=bpy.context.preferences.addons['cycles'].preferences; p.compute_device_type='OPTIX'; "
       "p.get_devices(); [setattr(d,'use',d.type!='CPU') for d in p.devices]; "
       "bpy.context.scene.cycles.device='GPU'; print('GPU', [d.name for d in p.devices if d.use])")
os.makedirs("/content/images", exist_ok=True)
paquet = int(globals().get("PAQUET", 20))
try:
    from google.colab import files as _files
except ImportError:
    _files = None

for i in range(0, len(images), paquet):
    lot = images[i:i + paquet]
    t = time.time()
    for f in lot:
        sh(f'{B} -b {F} --python-expr "{gpu}" -o /content/images/img_#### -F PNG -f {f} 2>&1 | grep -E "GPU|Saved|Time:|Error" || true')
    print(f"lot {lot[0]}-{lot[-1]} : {(time.time() - t) / len(lot):.0f} s par image", flush=True)
    z = f"/content/images_{lot[0]:04d}-{lot[-1]:04d}.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in lot:
            p = f"/content/images/img_{f:04d}.png"
            if os.path.exists(p):
                zf.write(p, os.path.basename(p))
    if _files:
        _files.download(z)
print("FINI", len(glob.glob("/content/images/*.png")), "images")
