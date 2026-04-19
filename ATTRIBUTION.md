# Attribution And Upstream Notice

This repository is derived from a minimal subset of the RIGNO project:

- Upstream repository: <https://github.com/camlab-ethz/rigno>
- RIGNO paper: <https://arxiv.org/abs/2501.19205>

The current repository is a modified research fork focused on a 3D IC
steady-state heat workflow. It is not an official RIGNO release.

## Retained Upstream Components

Only the core components needed by the Heat3D workflow are retained:

- Region interaction graph operator and graph builder logic
- Typed graph data structures and graph network components
- Operator input structures and shared model utilities
- Local support code required by the retained graph/model implementation

The original upstream generic CLI workflow for benchmark `.nc` datasets has
been removed from the public workflow.

## Heat3D-Specific Additions

This fork adds the Heat3D dataset adapter, 3D graph builder wrapper, steady heat
training/evaluation pipeline, and public entry scripts under `scripts/`.

The Heat3D changes adapt the retained RIGNO core to a simplified steady heat
operator:

```text
[thermal conductivity k(x), heat source q(x)] -> temperature T(x)
```

## License

The upstream project uses the MIT License. The upstream license text is retained
in `LICENSE` and must remain available in copies or substantial portions of the
software.

The original copyright notice in `LICENSE` is:

```text
Copyright (c) 2025 Computational and Applied Mathematics Laboratory @ ETH Zurich
```

Any public release of this fork should retain the MIT license text and this
upstream notice.

## Citation

Please cite RIGNO when using the retained RIGNO model core:

```bibtex
@inproceedings{mousavi2025rigno,
  title         = {RIGNO: A Graph-based framework for robust and accurate operator learning for PDEs on arbitrary domains},
  author        = {Sepehr Mousavi and Shizheng Wen and Levi Lingsch and Maximilian Herde and Bogdan Raonic and Siddhartha Mishra},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {38},
  year          = {2025}
}
```

If this Heat3D fork is used in a publication, cite both the upstream RIGNO work
and the publication or repository record for this fork once available.
