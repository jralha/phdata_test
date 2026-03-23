# Model Rollout, Versioning, and Scaling Strategy

## Goals

- Track model performance across experiments and data versions.
- Promote models through staging and production.
- Deploy model updates with no or minimal downtime.
- Scale inference capacity automatically.

## MLflow for Model Versioning

### Experiment and Performance Tracking

Use MLflow Tracking to log each training run from `create_model.py`.

Log per run:
- Parameters: algorithm, feature set, random seed, HPO search space, best params.
- Metrics: train and holdout RMSE, MAE, R2, MAPE, WAPE, and percent-within-threshold metrics.
- Artifacts: `model.pkl`, `model_features.json`, `input_feature_defaults.json`, `test_metrics.json`, and `hpo_results.json`.
- Tags: git commit SHA, dataset snapshot id, environment, trainer version.

### Registry and Stage Flags

Use MLflow Model Registry with named model versions and stage transitions:
- `None` or `Archived`: old or experimental versions.
- `Staging`: candidate model validated by offline metrics and smoke tests.
- `Production`: active serving model.

Staging and Production flags don't need to be exclusive as well.

Keeping different model versions online at the same time allows for gradual rollouts and/or A/B testing.

### Code updates:
- Training no longer needs to be handled here, this allows data scientists to maintain training flows in other repos.
- Requires extra validation to make sure the new incoming model still respects endpoint constraints (features, model libraries, etc.)
- Inference code needs to be updated to load the model artifact from mlflow's model registry instead of local.

### Promotion policy example:
1. Training pipeline registers a new model version.
2. CI runs integration tests and metric checks.
3. If quality gates pass, move to `Staging`.
4. Run tests in staging environment.
5. Approver promotes to `Production`. But current production endpoint stays online for easy rollbacks if needed.

### Quality Gates

Define explicit promotion thresholds to prevent regressions:
- Model metrics: Validate new incoming prod model performs better than current prod model, within a certain threshold.
- Engineering metrics: Implement load test and check latency and error rates to ensure new model doesn't break the endpoint.


## Scaling Endpoints

### SageMaker endpoints

- Real-time endpoint for online inference.
- Multi-model endpoint if serving many models with lower per-model traffic. Serverless endpoints a possibility if traffic is low and cost is a concern.
- Async or batch transform for large offline workloads.

### Auto Scaling Policies

- Target tracking by `InvocationsPerInstance` to maintain desired load.
- Scale out aggressively on high traffic.
- Scale in conservatively with cooldown windows to reduce thrash.
- Add CloudWatch alarm policies on p95 latency for protection.

## Observability

Monitor:
- EndpointInvocation metrics and errors.
- Latency percentiles.
- Container CPU and memory.
- **Data and prediction drift.** Automatic retraining?

Use alerts tied to SLOs and automatic rollback triggers where possible.
