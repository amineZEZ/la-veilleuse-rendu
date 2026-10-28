#!/bin/bash
# Rendu sur un Codespace (processeur) : images de DE à A (dans cet ordre, ex. 610 500),
# poussées toutes les 3 images sur la branche images-codespace.
#   bash codespace/rendu_cpu.sh 610 500
set -u
DE=$1; A=$2
cd "$(dirname "$0")/.."
if [ ! -x /tmp/blender/blender ]; then
  sudo apt-get -qq update && sudo apt-get -qq install -y libxrender1 libxi6 libxkbcommon0 libsm6 libgl1 libxxf86vm1 libxfixes3 libegl1 xz-utils > /dev/null
  curl -fsSL https://github.com/sebbar-studio-films/la-veilleuse-rendu/releases/download/v1/blender-5.2.2-linux-x64.tar.xz -o /tmp/b.tar.xz
  mkdir -p /tmp/blender && tar -xf /tmp/b.tar.xz -C /tmp/blender --strip-components=1 && rm /tmp/b.tar.xz
fi
[ -f /tmp/film.blend ] || curl -fsSL https://github.com/sebbar-studio-films/la-veilleuse-rendu/releases/download/v1/la_veilleuse.blend -o /tmp/film.blend
git fetch -q origin images-codespace 2>/dev/null && git checkout -q -B images-codespace origin/images-codespace || git checkout -q -B images-codespace
mkdir -p images
PAS=$([ "$DE" -gt "$A" ] && echo -1 || echo 1)
n=0
for f in $(seq "$DE" "$PAS" "$A"); do
  p=$(printf "images/img_%04d.png" "$f")
  [ -f "$p" ] && continue
  /tmp/blender/blender -b /tmp/film.blend -o "$PWD/images/img_####" -F PNG -f "$f" 2>&1 | grep -E "Saved|Time:" 
  n=$((n+1))
  if [ $((n % 3)) -eq 0 ]; then git add images && git commit -qm "images $f" && git push -q -u origin images-codespace; fi
done
git add images && git commit -qm "images fin" ; git push -q -u origin images-codespace
echo FINI
