### lr updater path
${python_env_path}\Lib\site-packages\mmcv\runner\hooks\lr_updater.py

**Note: in hooks directory, there are also "checkpoint" "optimizer" "momentum_updater" "sampler_seed" etc.
If you want to learn relevant cfgs for details, check these.**

### My supplements

### My modification
1. mmdet/datasets/coco.py \
Add specific eval index if necessary, in path "mmdet/datasets/coco.py" \
Or you can directly modify the "{envs}/torch1.4/Lib/site-packages/pycocotools/cocoeval.py" \
If you don't know the path for module file, you can import it in python file and run it easily. \
Then you can complete your own evaluation index based in specifics task.
2. mmdet/apis/inference.py \
Add cpu mode load checkpoints method.



### Restrict the pred boxes size range
You can inherit and rewrite the "${python path}/envs/torch1.4/Lib/site-packages/pycocotools/cocoeval.py" in func:setDetParams
