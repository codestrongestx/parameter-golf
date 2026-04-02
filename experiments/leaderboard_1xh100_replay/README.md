# Leaderboard 1xH100 Replay

Purpose:

- replay current leaderboard-scale configs on `1xH100`
- measure how much of the reported `8xH100` quality appears portable to a single GPU
- separate cleanly portable neural improvements from evaluation paths that depend heavily on `8` ranks

Current replay targets:

- official merged top record: `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- strongest frontier neural branch: `pr-728`
- strongest frontier rescore branch: `pr-888`

Interpretation rule:

- if a config stays strong on `1xH100`, it is a plausible innovation source for the next mainline push
- if a config collapses on `1xH100`, inspect whether the gain came from multi-rank optimization rather than a generally stronger recipe
