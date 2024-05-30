python launch.py --config configs/dge.yaml --train --gpu 0 \
    trainer.max_steps=1500 system.prompt_processor.prompt="Make the skeleton on fire" \
    data.source="/work/minghao/SegAnyGAussians/data/horns/" \
    system.seg_prompt="skeleton" \
    system.guidance.guidance_scale=15 \
    system.gs_source="/work/minghao/SegAnyGAussians/output/horns/point_cloud/iteration_30000/scene_point_cloud.ply" \
    system.gs_lr_scaler=0.1 system.gs_final_lr_scaler=0.1 system.color_lr_scaler=1 \
    system.opacity_lr_scaler=0.1 system.scaling_lr_scaler=0.1 system.rotation_lr_scaler=0.1