# Experiment records and baseline status

`experiments/week_05_baseline/experiment_config.yaml` is a **planned, not run** record. Cassava is the proposed baseline dataset, ResNet is the existing planned family, and variant/training hyperparameters remain unset. The repository does not provide a training command or claim baseline results.

For each future experiment, track its configuration, purpose, source revision, environment, dataset version, manifest hashes, class mapping, preprocessing settings, seed, protocol and small measured metrics. Record status honestly (planned/running/completed/failed) and retain limitations/failure notes. Do not populate a results field until measured outputs exist.

Store large outputs under an experiment's `outputs/`, `checkpoints/` or `predictions/` directories, which are ignored; record external artifact locations/checksums in a small tracked record. Do not blanket-ignore experiments or data/manifests.

Before baseline training: finalize architecture and hyperparameters, implement training/evaluation, verify dataset availability, use the fixed classification manifests, and record environment. Before temporal modeling: validate biological identity and forecast/evaluation design. Before severity: validate organ-area denominator and grade definitions. Before early-risk prediction: define outcome labels, horizon and observation cutoff.
