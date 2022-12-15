conda create --name locs python=3.9
conda activate locs
conda install pytorch=1.10.0 torchvision=0.11.0 torchaudio=0.10.0 cudatoolkit=10.2 setuptools=59.5.0 -c pytorch
conda install matplotlib=3.3.2 -c conda-forge
conda install pyg=2.0.3 -c pyg -c conda-forge
conda install tensorboard=2.6.0 -c pytorch
pip install -e .
