# Enhancing Locality

## dataset setup
```bash
mkdir datasets && cd datasets/
pip install gdown
gdown 1nebbxjchrIjAjVMLybvkfJuR-nD-ZGwF
unzip -q cub200_augmented.zip
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -zxf CUB_200_2011.tgz --no-same-owner
mv attributes.txt CUB_200_2011
rm CUB_200_2011.tgz cub200_augmented.zip
export data_dir=$PWD
export dataset_root=$PWD
```

## Installation

```bash
git clone https://github.com/openai/CLIP.git
pip install ftfy regex tqdm
pip install -e CLIP/
pip install pandas opencv-python grad-cam torchmetrics

scp ~/Downloads/backbone-checkpoints-Label_free-CBM.zip root@193.69.10.2:/workspace/Label-free-cbm/

pip install gdown
git submodule update --init --recursive --remote

# Install d2 dependencies
pip install 'fvcore>=0.1.5,<0.1.6' 'pycocotools>=2.0.2' cloudpickle omegaconf timm gdown

# Install d2
git clone https://github.com/facebookresearch/detectron2.git

pip install -e ./detectron2 --no-build-isolation

git clone https://github.com/zijizhu/VLPart.git

# macOS
# CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" uv pip install -e ./detectron2 --no-deps --no-build-isolation
```
