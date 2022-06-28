python src/guesswhat/train/train_qgen_reinforce_2oracle_w_guesser.py \
     -data_dir data/ \ # your data dir here
     -exp_dir out/loop/ \
     -config config/looper/config.json \
     -img_dir data/ft_vgg_img \
     -crop_dir data/ft_vgg_crop \
     -networks_dir out/ \
     -oracle_identifier 9cf445ee5b79194f57161a43799d7760 \ # your coop oracle id here
     -qgen_identifier ec20a89aacb6edef515cfc0c5b0f0493 \ # your qgen id here
     -guesser_identifier faa9bf620f7a02d95f55b7a25baa72da \ # your guesser id here
     -deceptive_oracle_identifier 9cf445ee5b79194f57161a43799d7760 \ # your non-coop oracle id here
     -evaluate_all false \
     -store_games true \
     -no_thread 2 \
     -train_guesser \ # trains coop guesser
     -train_qgen \ # trains on cooperation (deception) id, by default
     -train_on_original_task # trains on object recog. too

python src/guesswhat/train/train_qgen_reinforce_2oracle_w_guesser.py \
     -data_dir data/ \
     -exp_dir out/loop/ \
     -config config/looper/config.json \
     -img_dir data/ft_vgg_img \
     -crop_dir data/ft_vgg_crop \
     -networks_dir out/ \
     -oracle_identifier 9cf445ee5b79194f57161a43799d7760 \
     -qgen_identifier ec20a89aacb6edef515cfc0c5b0f0493 \
     -guesser_identifier faa9bf620f7a02d95f55b7a25baa72da \
     -deceptive_oracle_identifier 9cf445ee5b79194f57161a43799d7760 \
     -evaluate_all false \
     -store_games true \
     -no_thread 2 \
     -train_guesser \ # trains coop guesser
     -train_qgen # trains on cooperation (deception) id, by default

python src/guesswhat/train/train_qgen_reinforce_2oracle_w_guesser.py \
     -data_dir data/ \
     -exp_dir out/loop/ \
     -config config/looper/config.json \
     -img_dir data/ft_vgg_img \
     -crop_dir data/ft_vgg_crop \
     -networks_dir out/ \
     -oracle_identifier 9cf445ee5b79194f57161a43799d7760 \
     -qgen_identifier ec20a89aacb6edef515cfc0c5b0f0493 \
     -guesser_identifier faa9bf620f7a02d95f55b7a25baa72da \
     -deceptive_oracle_identifier 9cf445ee5b79194f57161a43799d7760 \
     -evaluate_all false \
     -store_games true \
     -no_thread 2 \
     -train_guesser \ # trains coop guesser
     -train_qgen \
     -train_only_on_task # trains on object recog. only

python src/guesswhat/train/train_qgen_reinforce_2oracle_w_guesser.py \
     -data_dir data/ \
     -exp_dir out/loop/ \
     -config config/looper/config.json \
     -img_dir data/ft_vgg_img \
     -crop_dir data/ft_vgg_crop \
     -networks_dir out/ \
     -oracle_identifier 9cf445ee5b79194f57161a43799d7760 \
     -qgen_identifier ec20a89aacb6edef515cfc0c5b0f0493 \
     -guesser_identifier faa9bf620f7a02d95f55b7a25baa72da \
     -deceptive_oracle_identifier 9cf445ee5b79194f57161a43799d7760 \
     -evaluate_all false \
     -store_games true \
     -no_thread 2 \
     -train_guesser # trains only coop. guesser, i.e., supervised learning of qgen

