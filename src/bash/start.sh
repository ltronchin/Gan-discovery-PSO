#!/bin/bash

# Autoencoder Training for evaluation
# python3 ./src/training/cae.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist
# Classifier Training for evaluation
# python3 ./src/training/classifiers.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --path_latent_den_cae "./data/interim/mnist/00001--cae.py"
# CNN Training for PSO
# python3 ./src/training/cnn.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:1 --dataset mnist
# python3 ./src/training/cnn_multipatient.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:1 --dataset mnist

latent_dim_list=(2 3 4 6 8 10 20 30 100)
idd_list=(1 2 3 4 5 6 7 8 9)
for id in 1 2 3 4 5 6 7 8 9
do
  # Gan Training
  #python3 ./src/training/dcgan.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp ${idd_list[id-1]} --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_den_cae "./models/mnist/00001--cae.py"  --path_classifiers "./models/mnist/00001--classifiers.py"
  # PSO Discovery
  #python3 ./src/training/pso_discovery.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_cnn "./models/mnist/00001--cnn_multipatient.py"  --path_gan "./models/mnist/0000${idd_list[id-1]}--dcgan.py" --w_ine 0.73 --w_cogn 1.496 --w_soci 1.496
  #python3 ./src/training/pso_discovery.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 2 --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_cnn "./models/mnist/00001--cnn_multipatient.py"  --path_gan "./models/mnist/0000${idd_list[id-1]}--dcgan.py" --w_ine 1.0 --w_cogn 2.0 --w_soci 2.0 --schedule_ine True
  #python3 ./src/training/pso_discovery.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 3 --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_cnn "./models/mnist/00001--cnn_multipatient.py"  --path_gan "./models/mnist/0000${idd_list[id-1]}--dcgan.py" --w_ine 0.5 --w_cogn 0.8 --w_soci 0.9

  # Inversion
  #python3 ./src/training/inverter.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_cnn "./models/mnist/00001--cnn_multipatient.py"  --path_gan "./models/mnist/0000${idd_list[id-1]}--dcgan.py" --inverter_train_fun "pix_rec"
  #python3 ./src/training/inverter.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 2 --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_cnn "./models/mnist/00001--cnn_multipatient.py"  --path_gan "./models/mnist/0000${idd_list[id-1]}--dcgan.py" --inverter_train_fun "pix_fea_rec_adv"

  # PSO analysis
  # Extraction of the latent ood and iid representation from the inverter
  python3 ./src/training/ood_extractor.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_inverter "./models/mnist/00002--inverter.py"  --path_gan "./models/mnist/0000${idd_list[id-1]}--dcgan.py"
  python3 ./src/training/iid_extractor.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_inverter "./models/mnist/00002--inverter.py"  --path_gan "./models/mnist/0000${idd_list[id-1]}--dcgan.py"

  # Analysis # todo check the pso_analysis_clustering.py and merge with pso_inverter_analysis.py
  python3 ./src/training/pso_analysis_clustering.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_inverter "./models/mnist/00002--inverter.py"  --path_gan "./models/mnist/0000${idd_list[id-1]}--dcgan.py" --path_iid_pso_discovery "./data/interim/00001--pso_discovery.py" --ood_analysis "ood"
  python3 ./src/training/pso_analysis_clustering.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 2 --gpu cuda:0 --dataset mnist --latent_dim "${latent_dim_list[id-1]}" --path_inverter "./models/mnist/00002--inverter.py"  --path_gan "./models/mnist/0000${idd_list[id-1]}--dcgan.py" --path_iid_pso_discovery "./data/interim/00001--pso_discovery.py" --ood_analysis "iid"

  sleep 5m
done