# V6 hard/OOD preregistration

Status: frozen before hard-specific subgroup metrics.

The canonical P1h dataset has no registered OOD role. Earlier P1e OOD roles use a different dataset and support, so they are not silently reused here.

The only executable stress role is `hard_input_stress_corner_v1`: 16 test-holdout cases with top h=1000 W/(m²·K), bottom h=20 W/(m²·K), and package power=6 W. Selection uses only the frozen input case table and manifest, never target temperature or model error.

The underlying test labels were already opened in the corrected confirmatory holdout. This gate freezes the subgroup before any hard-specific metric is computed; it cannot retroactively claim that the physical labels were never read.

4096 is default/hotspot-oriented, 8192 is balanced full-field, and 16384 is the maximum full-field accuracy mode. 32768 is excluded.

No hard/OOD result may change model, checkpoint, resolution, graph, reconstruction, or any later tuning decision.
