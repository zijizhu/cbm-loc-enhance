# Enhancing Locality

## dataset setup
```
pip install gdown
gdown 1nebbxjchrIjAjVMLybvkfJuR-nD-ZGwF
unzip -q cub200_augmented.zip
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -zxf CUB_200_2011.tgz --no-same-owner
mv attributes.txt CUB_200_2011
```

## Installation

```bash
pip install gdown
git submodule update --init --recursive --remote

# Install d2 dependencies
uv pip install 'fvcore>=0.1.5,<0.1.6' 'pycocotools>=2.0.2' cloudpickle omegaconf timm gdown

# Install d2
git clone https://github.com/facebookresearch/detectron2/tree/main

pip install -e ./detectron2

# macOS
# CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" uv pip install -e ./detectron2 --no-deps --no-build-isolation

mv attributes.txt CUB_200_2011
```