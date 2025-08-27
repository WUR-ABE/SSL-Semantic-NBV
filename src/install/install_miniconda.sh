wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
bash Miniconda3-py38_4.12.0-Linux-x86_64.sh
rm Miniconda3-py38_4.12.0-Linux-x86_64.sh
eval "$(cat ~/.bashrc | tail -n +10)"
conda deactivate
conda config --set auto_activate_base false
conda create --name av python=3.8.10