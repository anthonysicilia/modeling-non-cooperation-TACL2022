# coop oracle
python src/guesswhat/train/train_oracle_dim.py \
   -data_dir data \ # your data dir here
   -img_dir data/img/ft_vgg_img \
   -crop_dir data/img/ft_vgg_crop \
   -config config/oracle/original.json \
   -exp_dir out/oracle \
   -non_deceptive

# all the non-coop bad oracles
# data dir here should be different from coop oracle
# or, you can also use the custom dataset argument
python src/guesswhat/train/train_oracle_dim.py \
   -data_dir data \ 
   -img_dir data/img/ft_vgg_img \
   -crop_dir data/img/ft_vgg_crop \
   -config config/oracle/original.json \
   -exp_dir out/oracle

python src/guesswhat/train/train_oracle_dim.py \
   -data_dir data \
   -img_dir data/img/ft_vgg_img \
   -crop_dir data/img/ft_vgg_crop \
   -config config/oracle/with_history.json \
   -exp_dir out/oracle

python src/guesswhat/train/train_oracle_dim.py \
   -data_dir data \
   -img_dir data/img/ft_vgg_img \
   -crop_dir data/img/ft_vgg_crop \
   -config config/oracle/with_image.json \
   -exp_dir out/oracle

python src/guesswhat/train/train_oracle_dim.py \
   -data_dir data \
   -img_dir data/img/ft_vgg_img \
   -crop_dir data/img/ft_vgg_crop \
   -config config/oracle/with_both.json \
   -exp_dir out/oracle