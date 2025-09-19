# Enhancing Locality

## dataset setup
```bash
# cd into data roor directory
pip install gdown
gdown 1nebbxjchrIjAjVMLybvkfJuR-nD-ZGwF
unzip -q cub200_augmented.zip
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -zxf CUB_200_2011.tgz --no-same-owner
mv attributes.txt CUB_200_2011
rm CUB_200_2011.tgz cub200_augmented.zip
export datadir=$PWD
export datasetroot=$PWD
# cd into source code directory
```

## Dependency Installation

```bash
# Install dependencies
pip install torch torchvision pandas opencv-python grad-cam timm torcheval
git clone https://github.com/openai/CLIP.git
pip install ftfy regex tqdm
pip install -e CLIP/

# Install d2: https://stackoverflow.com/a/79095245/17662217
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# cd into cbm-loc/enhance
rmdir VLPart
git clone https://github.com/zijizhu/VLPart.git
```
## Training

```sh

python3 main.py \
    --data-dir $datadir \
    --log-dir $logdir \
    --name 'densenet161-full' \
    --evaluate
evaluate
```
