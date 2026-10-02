# Rigging

## Rigging Generation

### Generate rigging using UniRig
```bash
# clone UniRig repo in this diretory
bash run_unirig.sh /path/to/filename.glb  # The script generates filename_skin.fbx
# generate rig file
blender --background --python blender_read_fbx.py -- --filepath /path/to/filename.glb
# textured mesh
# ...
```


