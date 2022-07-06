CKPT=/ckpts/stylegan2-ffhq-256x256.pkl
DATA=datasets/ir_faces/
python train.py --outdir=training-runs --data=$DATA --resume ffhq256 --gpus=8 --kimg 1000