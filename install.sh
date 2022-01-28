conda create --name locs python=3.9
conda activate locs
conda install pytorch torchvision torchaudio cudatoolkit=10.2 setuptools=59.5.0 -c pytorch
conda install matplotlib -c conda-forge
conda install pyg -c pyg -c conda-forge
conda install fvcore iopath pytorch3d -c fvcore -c iopath -c pytorch3d
conda install kornia -c conda-forge
conda install tensorboard -c pytorch
pip install -e .
