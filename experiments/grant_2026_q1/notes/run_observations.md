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
- `1x H100` should be the default iteration machine now
- `8x H100` should be reserved for the final verification attempt
