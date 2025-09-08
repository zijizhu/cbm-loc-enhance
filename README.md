# Enhancing Locality

## Installation

```bash
# Install d2 dependencies
uv pip install 'fvcore>=0.1.5,<0.1.6' 'pycocotools>=2.0.2' cloudpickleomegaconf timm
# Install d2
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" uv pip install -e ./detectron2 --no-deps --no-build-isolation
```