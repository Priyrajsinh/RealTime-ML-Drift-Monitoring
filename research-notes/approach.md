# Research Notes — B5 Real-Time ML + Drift Monitoring

## Papers I Read Before Starting
- Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift (Rabanser et al., 2019)
- Population Stability Index (PSI) — standard banking metric for distribution shift
- Kolmogorov-Smirnov test — non-parametric two-sample test
- Evidently AI documentation — open-source ML monitoring framework

## Architecture Decisions
- RandomForest baseline: strong default, fast training, SHAP-compatible (TreeExplainer)
- PSI for drift detection: industry standard in banking/finance, threshold 0.2 well-established
- KS test as second signal: catches shape changes PSI might miss
- Prometheus + Grafana: industry standard observability stack
- Alertmanager: production alerting (PSI > 0.2 fires alert)
- Data drift vs concept drift: distinguished for interview depth
- SHAP under drift: ties back to B2, shows portfolio coherence

## Surprising Findings
- [Fill in after building]
