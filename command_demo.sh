# CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path logs/log_kn/checkpoint.tar
# my case:
CUDA_VISIBLE_DEVICES=0 python demo/demo.py --checkpoint_path logs/realsense/checkpoint_epoch7.tar
cd demo
python vis.py