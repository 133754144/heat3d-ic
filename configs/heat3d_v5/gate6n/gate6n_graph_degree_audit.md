# Gate 6N P5 graph degree audit

- baseline: `V4P5_38_gate6n_v36_r2r_mask_p005_e600`
- train samples: 672
- unique normalized-coordinate topologies: 1

- topology `93b2d9aad02d`: 672 samples, r2r min degree=5
  - p=0.02: max zero-degree=0, same-seed reproducible=True
  - p=0.05: max zero-degree=0, same-seed reproducible=True
  - p=0.10: max zero-degree=0, same-seed reproducible=True
  - p=0.20: max zero-degree=0, same-seed reproducible=True
  - p=0.50: max zero-degree=2, same-seed reproducible=True
  - exact e600 p=0.02: masks=14400, max zero-in/out=0/0, max isolated=0, max weak components=1, all safe=True
  - exact e600 p=0.05: masks=14400, max zero-in/out=0/0, max isolated=0, max weak components=1, all safe=True
