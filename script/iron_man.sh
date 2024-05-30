python launch.py --config configs/dge.yaml --train --gpu 0 \
    trainer.max_steps=1500 system.prompt_processor.prompt="Turn him into iron man" \
    data.source="/work/minghao/SegAnyGAussians/data/person-small/" \
    data.use_original_resolution=True \
    system.gs_source="/work/minghao/SegAnyGAussians/output/person-small/point_cloud/iteration_30000/scene_point_cloud.ply"