------ GAN TRAINING -----
1) [LOCALE] creare il file "requirements.txt". Spostandosi della cartella dove e' presente il codice "src", da command line accedi all'envinroments in uso
(vedi note) ed esegui il comando "pipreqs --encoding=utf8 .\". Se negli script in "src" non ci sono errori verrà generato un file "requirements".

2) [WORKSTATION] Creare uno screen usando un nome descrittivo con il comando "screen -S your_session_name"

3) [WORKSTATION>SCREEN][SE NON ESISTE L'IMMAGINE>con il comando "docker images"] All'interno dello screen scaricare l'immagine del container: 
"docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime"

4) [WORKSTATION>SCREEN] Lanciare il container Torch
"docker run --gpus all --ipc=host -it --rm -v /data01/ltronchin/ltronchin/:/ltronchin pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime"

5) [WORKSTATION>SCREEN>CONTAINER] Aggiornare Ubuntu
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y

6) [WORKSTATION>SCREEN>CONTAINER] Installare i pacchetti necessari tramite il comando "pip install -r requirements.txt"

7) [WORKSTATION>SCREEN>CONTAINER] In base al file config di interesse lanciare: "python3 ./src/models/"....".py -f ./.yaml"

	Ad esempio:
	python3 ./src/models/dcgan_mnist.py -f ./configs/dcgan_mnist.yaml -g cuda:1 --id_run 0 --> se si vuole scegliere la GPU

	Se si vogliono lanciare una batteria di esperimenti si puo' usare il comando bash: "bash src/bash/"....".sh"

8) Tensorboard Torch
tensorboard --logdir=runs
runs = path to "logs" 

NOTE:
[LOCAL]>si lavora in locale
[WORKSTATION]>si lavora nel main environment della macchina
[WORKSTATION-SCREEN]>si lavora all'interno dello screen creato
[WORKSTATION>SCREEN>CONTAINER]>si lavora all'interno dello screen e all'interno dell container

LOCAL Python environment: "torch_gpu_xai4covid"

NEL CASO DI ERRORE DI IMPORT DI MODULI CUSTOM VEDI:
https://towardsdatascience.com/how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c
Probabilmente serve aggiungere il riferimento al progetto> ' export PYTHONPATH="${PYTHONPATH}:/src" '

ERRORE: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y

Ordine di esecuzione
1) Training CAE
2) Training Classifiers 
3) Training GAN

Launchpad:

# CAE
python3 ./src/training/cae.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist

# Classifiers
python3 ./src/training/classifiers.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --path_latent_den_cae "./data/interim/mnist/00001--cae.py"                                                                                                                                                    

# GAN
python3 ./src/training/dcgan.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim 2 --path_den_cae "./models/mnist/00001--cae.py"  --path_classifiers "./models/mnist/00001--classifiers.py"

# CNN
python3 ./src/training/cnn.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:1 --dataset mnist   
python3 ./src/training/cnn_multipatient.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:1 --dataset mnist

# PSO discovery
python3 ./src/training/pso_discovery.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:1 --dataset mnist --latent_dim 2 --path_cnn "./models/mnist/00001--cnn_multipatient.py"  --path_gan "./models/mnist/00001--dcgan.py" --w_ine 0.73 --w_cogn 1.496 --w_soci 1.496
python3 ./src/training/pso_discovery.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 2 --gpu cuda:1 --dataset mnist --latent_dim 2 --path_cnn "./models/mnist/00001--cnn_multipatient.py"  --path_gan "./models/mnist/00001--dcgan.py" --w_ine 1.0 --w_cogn 2.0 --w_soci 2.0 --schedule_ine True
python3 ./src/training/pso_discovery.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 3 --gpu cuda:1 --dataset mnist --latent_dim 2 --path_cnn "./models/mnist/00001--cnn_multipatient.py"  --path_gan "./models/mnist/00001--dcgan.py" --w_ine 0.5 --w_cogn 0.8 --w_soci 0.9

python3 ./src/training/pso_discovery.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 4 --gpu cuda:0 --dataset mnist --latent_dim 2 --path_cnn "./models/mnist/00001--cnn_multipatient.py"  --path_gan "./models/mnist/00001--dcgan.py" --w_ine 1.0 --w_cogn 2.0 --w_soci 2.0 --schedule_ine True

# INVERTER

# PSO ANALYSIS
python3 ./src/training/pso_analysis_clustering.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim 10 --path_inverter "./models/mnist/00002--inverter.py"  --path_gan "./models/mnist/00006--dcgan.py" --path_iid_pso_discovery "./data/interim/00004--pso_discovery.py" --ood_analysis "ood"
python3 ./src/training/pso_analysis_clustering.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 2 --gpu cuda:0 --dataset mnist --latent_dim 10 --path_inverter "./models/mnist/00002--inverter.py"  --path_gan "./models/mnist/00006--dcgan.py" --path_iid_pso_discovery "./data/interim/00004--pso_discovery.py" --ood_analysis "iid"

# PSO INVERTER
python3 ./src/training/pso_inverter.py --cfg_file ./configs/dcgan_mnist.yaml --id_exp 1 --gpu cuda:0 --dataset mnist --latent_dim 10 --path_cnn "./models/mnist/00001--cnn_multipatient.py" --path_gan "./models/mnist/00006--dcgan.py" --path_inverter "./models/mnist/00002--inverter.py" --path_ood_patient 5 --w_ine 0.73 --w_cogn 1.496 --w_soci 1.496 --optimize_in_training True


# BASH
bash ./src/bash/start.sh

# BASH PSO INVERTER
bash ./src/bash/start_pso_optimize.sh
