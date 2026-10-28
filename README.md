<div align="center">

# HierarchicalKV for HCU

</div>

## Open Source Attribution
This repository is based on [NVIDIA Merlin HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV.git) and contains some changes for HCU/ROCm environments.

- Direct upstream: [NVIDIA Merlin HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV.git)
- Upstream branch: master
- Upstream baseline commit: 9c197a9c558d1e8285c2e50c1974f0f102826f11
- Third-party inventory: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- License: Apache-2.0
- Upstream-derived historical reference: [README_ORIGIN.md](README_ORIGIN.md)

## Modification Scope
Adjusted some implementation logic due to warpSize differences between HCU/ROCm and CUDA. Unless explicitly stated otherwise, upstream-originated files remain governed by their inherited Apache-2.0 license terms.

Modified by Hygon Information Technology Co., Ltd.

## Testing and Accuracy

### Build the tests
```bash
cd HierarchicalKV
git submodule update --init --recursive
cmake ../
make -j32
```
Run the public HierarchicalKV test suite on an HCU device:
```bash
cd HierarchicalKV/build
# Execute the executable program.
```

## References

- [HierarchicalKV direct upstream](https://github.com/NVIDIA-Merlin/HierarchicalKV.git)
- [Upstream-derived historical reference](README_ORIGIN.md)

## License
Apache-2.0 License
Additional modifications copyright: Copyright 2026 Hygon Information Technology Co., Ltd.
