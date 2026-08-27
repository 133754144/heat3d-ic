# V6 dataset Hugging Face sync

本次同步只覆盖已经实际用于 V6 训练与性能评估的数据集：

- `heat3d_v6_p1h_shared_support1024_v0`：唯一默认 V6-layer canonical
  dataset。HF 上已有完整 subset，本次核验其 manifest、240,825-node
  full-field archive 和 sample tree，未重复上传或覆盖。
- `heat3d_v6_p1i_continuous_physics1024_v1`：P1i formal V6 random-block
  training/evaluation family。HF 原先只有 full-field/sidecar；本次从
  devbox 可续传复制 1,024 个 sample directories，补入 canonical manifest
  与 subset README，并保留已有 full-field archive。

## Remote placement

Both subsets remain parallel under
`subsets/<dataset_id>/` in
`133754144X/heat3d-thermal-simulation`:

- `subsets/heat3d_v6_p1h_shared_support1024_v0/`
- `subsets/heat3d_v6_p1i_continuous_physics1024_v1/`

The metadata/index update and the missing P1i payload were committed to HF as
`20934dcd79911b80287ae885749298371b9850ad`.

## Integrity and roles

| dataset | cases / split | manifest SHA256 | full-field SHA256 | status |
| --- | --- | --- | --- | --- |
| P1h shared support | 1024 / 768 train, 128 valid, 128 test | `324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5` | `f58141b3f365c5c90a57ec3802ae57c7e7afbf83ba0ab988060a617164b14c00` | verified existing canonical |
| P1i continuous random-block | 1024 / 768 train, 128 valid, 128 test | `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514` | `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb` | completed HF subset |

P1i additionally has sidecar SHA256
`4dc526c75aff4de702482f87c969ed3e427cf882c42444fb976917ab88a1a130`.
Its 9,216 sample payload files were checked against the frozen manifest after
the devbox transfer with zero mismatches. The HF tree contains 1,024 sample
directories and the expected full-field metadata. P1h's downloaded HF manifest
is byte-identical to the local frozen manifest.

`test_iid` remains a post-freeze confirmatory role and is not a training or
checkpoint-selection source. Sealed IID is not uploaded or opened by this
operation. Historical V6 subsets are retained at their existing paths.

The machine-readable record for this sync is
`configs/heat3d_v6/v6_hf_dataset_sync_manifest.json`.
