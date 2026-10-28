"""Rendu de « La veilleuse » sur les cartes graphiques de Modal : une image par conteneur, en parallèle.

    ../image-to-3d/.venv/bin/modal run modal/rendu_modal.py --images 100,680          # test
    ../image-to-3d/.venv/bin/modal run modal/rendu_modal.py --images 1-840 --gpu L4   # film entier

Les images reviennent directement sur le Mac, dans la-veilleuse/rendus/final/.
"""
import os
import pathlib
import time

import modal

BLEND_URL = "https://github.com/sebbar-studio-films/la-veilleuse-rendu/releases/download/v1/la_veilleuse.blend"
BLENDER_URL = "https://github.com/sebbar-studio-films/la-veilleuse-rendu/releases/download/v1/blender-5.2.2-linux-x64.tar.xz"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "xz-utils", "libxrender1", "libxi6", "libxkbcommon0", "libsm6", "libgl1",
                 "libxxf86vm1", "libxfixes3", "libegl1")
    .run_commands(
        f"curl -fsSL {BLENDER_URL} -o /tmp/b.tar.xz && mkdir -p /blender && tar -xf /tmp/b.tar.xz -C /blender --strip-components=1 && rm /tmp/b.tar.xz",
        f"curl -fsSL {BLEND_URL} -o /film.blend",
    )
    .add_local_file(pathlib.Path(__file__).parent.parent / "colab" / "gpu.py", "/gpu.py")
)
app = modal.App("la-veilleuse", image=image)
GPU = os.environ.get("MODAL_GPU", "L4")


@app.function(gpu=GPU, timeout=1800, max_containers=int(os.environ.get("MODAL_MAX", "50")), retries=1)
def rendre(f: int) -> bytes:
    import subprocess
    t = time.time()
    r = subprocess.run(["/blender/blender", "-b", "/film.blend", "--python", "/gpu.py",
                        "-o", "/tmp/img_####", "-F", "PNG", "-f", str(f)], capture_output=True, text=True)
    p = f"/tmp/img_{f:04d}.png"
    if not os.path.exists(p):
        raise RuntimeError(f"image {f} : échec\n" + (r.stdout + r.stderr)[-2000:])
    print(f"image {f} : {time.time() - t:.0f} s")
    return open(p, "rb").read()


@app.local_entrypoint()
def main(images: str = "100", sortie: str = ""):
    liste = []
    for part in images.split(","):
        a, _, b = part.partition("-")
        liste += list(range(int(a), int(b or a) + 1))
    out = pathlib.Path(sortie or pathlib.Path(__file__).resolve().parents[2] / "la-veilleuse" / "rendus" / "final")
    out.mkdir(parents=True, exist_ok=True)
    liste = [f for f in liste if not (out / f"img_{f:04d}.png").exists()]   # reprise : on saute ce qui existe
    print(f"{len(liste)} images à calculer sur {GPU}")
    t = time.time()
    for f, data in zip(liste, rendre.map(liste, order_outputs=True, return_exceptions=True)):
        if isinstance(data, Exception):
            print("ÉCHEC", f, str(data)[:300])
            continue
        (out / f"img_{f:04d}.png").write_bytes(data)
    print(f"fini en {time.time() - t:.0f} s")
