python launch.py --config configs/dge.yaml --train --gpu 0 \
    trainer.max_steps=1500 system.prompt_processor.prompt="Make the person wear clothes with a pineapple pattern" \
    data.source="/work/minghao/SegAnyGAussians/data/person-small/" \
    system.color_lr_scaler=1 \
    system.gs_source="/work/minghao/SegAnyGAussians/output/person-small/point_cloud/iteration_30000/scene_point_cloud.ply"