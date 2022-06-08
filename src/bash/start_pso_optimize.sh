#!/bin/bash

for p_id in 0 1 2 3 4 5 6
do
  # PSO optimization process
  python3 ./src/training/pso_inverter.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp ${p_id} --gpu cuda:0 --dataset mnist --latent_dim 10 --path_cnn "./models/mnist/00001--cnn_multipatient.py" --path_gan "./models/mnist/00006--dcgan.py" --path_inverter "./models/mnist/00002--inverter.py" --path_ood_patient ${p_id} --w_ine 0.73 --w_cogn 1.496 --w_soci 1.496 --control_pso_fitness 'optimize_in_training'
  python3 ./src/training/pso_inverter.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp ${p_id} --gpu cuda:0 --dataset mnist --latent_dim 10 --path_cnn "./models/mnist/00001--cnn_multipatient.py" --path_gan "./models/mnist/00006--dcgan.py" --path_inverter "./models/mnist/00002--inverter.py" --path_ood_patient ${p_id} --w_ine 0.73 --w_cogn 1.496 --w_soci 1.496 --control_pso_fitness 'optimize_out_training'

  sleep 2m

  # PSO Analysis
  python3 ./src/training/pso_inverter_analysis.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim 10  --path_gan "./models/mnist/00006--dcgan.py" --path_inverter "./models/mnist/00002--inverter.py"  --path_iid_pso_discovery "./data/interim/00001--pso_discovery.py" --path_ood_pso_inverter "./data/interim/0000${p_id}--pso_inverter.py/optimize_in_training" --path_ood_patient ${p_id}
  python3 ./src/training/pso_inverter_analysis.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim 10  --path_gan "./models/mnist/00006--dcgan.py" --path_inverter "./models/mnist/00002--inverter.py"  --path_iid_pso_discovery "./data/interim/00001--pso_discovery.py" --path_ood_pso_inverter "./data/interim/0000${p_id}--pso_inverter.py/optimize_out_training" --path_ood_patient ${p_id}

  sleep 2m
done


