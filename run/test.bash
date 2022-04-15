name=reverie-test
flag="--description test_submit

      --train validlistener
      --features places365
      --load load_test/best_val_unseen
      --submit

      --test_only 0
      --mode test
      --maxAction 15
      --maxInput 50
      --batchSize 2
      --feedback sample
      --lr 1e-5
      --iters 200000
      --optim adamW

      --attn soft
      --mlWeight 0.20
      --featdropout 0.4
      --angleFeatSize 128
      --subout max
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python r2r_src/train.py $flag --name $name
