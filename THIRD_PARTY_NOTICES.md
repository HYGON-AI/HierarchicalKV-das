# Third-Party Notices

This file records the third-party projects directly referenced by this source tree. Their original copyright and license notices remain in effect.

## HierarchicalKV

- Project: NVIDIA Merlin HierarchicalKV
- Repository: https://github.com/NVIDIA-Merlin/HierarchicalKV.git
- Branch: master
- Baseline commit: 9c197a9c558d1e8285c2e50c1974f0f102826f11
- License: Apache-2.0
- Modifications: Adaptations for HCU/ROCm

The unmodified upstream README at the fixed baseline is retained in [README_ORIGIN.md](README_ORIGIN.md). The upstream license text is retained in [LICENSE](LICENSE).

## googletest

- Project: googletest
- Repository: https://github.com/google/googletest.git
- Branch: 1.13.x
- Baseline commit: 0e0d9feefab1b51aaab9dfd70132e93c0b6964e5
- License: BSD 3-Clause
## ROCm libhipcxx

- Project: ROCm libhipcxx
- Repository: https://github.com/ROCm/libhipcxx.git
- Branch: release/2.2.x
- Baseline commit: branch tip at release time (pin the exact commit before release)
- License: Apache-2.0
- Local paths: `include/cuda/`, `include/hip/`, `include/nv/`
- Modifications: Integrated for HCU/ROCm compatibility; original copyright and license notices are retained.

## NVIDIA Merlin HierarchicalKV test sources

- Project: NVIDIA Merlin HierarchicalKV
- Repository: https://github.com/NVIDIA-Merlin/HierarchicalKV.git
- Branch: master
- Baseline commit: 9c197a9c558d1e8285c2e50c1974f0f102826f11
- License: Apache-2.0
- Local paths: `tests/**/*.hip`
- Modifications: Source files were converted from the upstream `*.cu` filenames to `*.hip`; record any additional HYGON code changes separately.
