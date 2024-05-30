python launch.py --config configs/dge.yaml --train --gpu 0 \
    trainer.max_steps=1500 system.prompt_processor.prompt="Turn him into a werewolf" \
    data.source="/work/minghao/SegAnyGAussians/data/face/" \
    system.guidance.guidance_scale=12.5 \
    system.gs_source="/work/minghao/SegAnyGAussians/output/face/point_cloud/iteration_30000/scene_point_cloud.ply"