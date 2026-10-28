# Active la carte graphique (OptiX, sinon CUDA) pour Cycles, puis laisse Blender calculer.
import bpy

p = bpy.context.preferences.addons["cycles"].preferences
for kind in ("OPTIX", "CUDA"):
    try:
        p.compute_device_type = kind
        p.get_devices()
        gpus = [d for d in p.devices if d.type == kind]
        if gpus:
            break
    except Exception as e:
        print("pas de", kind, e)
for d in p.devices:
    d.use = d.type != "CPU"
bpy.context.scene.cycles.device = "GPU"
print("GPU :", p.compute_device_type, [d.name for d in p.devices if d.use], flush=True)
