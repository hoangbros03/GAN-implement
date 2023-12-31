python cnn_gan/train.py \
    --epochs 200 \
    --image_size 64 \
    --batch_size 32 \
    --noise_size 100 \
    --learning_rate 0.0001 \
    --proportion 3 \
    --discriminator_type normal \
    --generator_type normal \
    --device cpu \
    --data_root data \
    --output_dir output_models \
    --save_model \
    --key None \
    --num_workers 2 \
    --save_frequency 3 \
    --channel 3