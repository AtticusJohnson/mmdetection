### lr updater path
${python_env_path}\Lib\site-packages\mmcv\runner\hooks\lr_updater.py

**Note: in hooks directory, there are also "checkpoint" "optimizer" "momentum_updater" "sampler_seed" etc.
If you want to learn relevant cfgs for details, check these.**

### My modification
1. Add specific eval index if necessary, in path "mmdet/datasets/coco.py"
The you can complete your own evaluation index based in specifics task.

### Restrict the pred boxes size range
You can inherit and rewrite the "${python path}/envs/torch1.4/Lib/site-packages/pycocotools/cocoeval.py" in func:setDetParams
