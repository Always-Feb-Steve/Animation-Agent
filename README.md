# VideoArticulation

## Environment Set up
Assuming CUDA 12.1, you can set up the environment using the following commands:
```
conda create -n VideoArticulation python=3.10
conda activate VideoArticulation
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/
```

Download SAM-2 checkpoints:
```
cp download_ckpts.sh <path-to-save-ckpts>
cd <path-to-save-ckpts>
bash <path-to-save-ckpts>/download_ckpts.sh
ln -s <path-to-save-ckpts> <path-to-this-repo>/checkpoints
```

## Unit Test Example
Camera pose optimization:
```
python test/camera_optim.py
```

Linear Blend Skinning:
```
python test/rigging_test.py
```

## Animation from Video
Use the following commands to animate a given 3d .obj with a given rigging configuration and a given reference video:
```
python rigging_video_optim.py --mesh_path <path-to-mesh> --rig_path <path-to-rig-file> --video_path <path-to-video>
```

For example:
```
python rigging_video_optim.py --mesh_path asset/deer_remesh_0410234018_texture_obj/deer_remesh_0410234018_texture.obj --rig_path asset/deer_remesh_0410234018_texture_obj/deer_ori_rig.txt --video_path asset/deer_remesh_0410234018_texture_obj/deer_animation.mp4
```
<!-- ## Use Segment-Anything 2 and Download Checkpoints

```
cp download_ckpts.sh /data/your_name/sam2
cd /data/yourname
mkdir sam2
bash /data/your_name/sam2/download_ckpt.sh
ln -s /data/your_name/ /home/your_name/VideoArticulation/checkpoints
``` -->

<!-- ## Use seg script to get mask video
```
cd utils
python seg.py --video video_path --frames frames_save_path --mask ground_truth_mask --threshold mask_threshold for binary
python seg.py \
  --video ../asset/deer_remesh_0410234018_texture_obj/deer_animation.mp4 \
  --frames ../asset/deer_remesh_0410234018_texture_obj/frames \
  --mask ../asset/deer_remesh_0410234018_texture_obj/render_silhouette.png \
  --threshold 0.8
``` -->