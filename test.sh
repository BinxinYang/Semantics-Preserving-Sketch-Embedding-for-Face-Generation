CUDA_VISIBLE_DEVICES=2 python scripts/inference.py \
--exp_dir=result \
--checkpoint_path=checkpoints/model.pt \
--data_path=examples/sketch \
--target_path=examples/appearance \
--test_batch_size=1 \
--couple_outputs \
--test_workers=1