# Run Observations

Use this file as the rolling human log for the campaign.

Suggested format for each block:

## YYYY-MM-DD

- question being asked
- exact config difference
- gpu used
- key metrics
- what changed next

Seed observations from prior work:

- cheap `3090` CUDA smoke is already proven in the earlier DepthFiLM branch
- this campaign should still use `3090` first to kill bad directions cheaply
- `1x H100` is for lanes that already look promising
- `8x H100` is not part of the active plan; it is a later user decision
