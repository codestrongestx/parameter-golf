Here’s the state of play.

The official merged leaderboard in the repo currently has #549 at 1.1194 BPB, then #414 at 1.1228, #315 at 1.1248, #287 at 1.1271, and #265 at 1.1307. To take the official crown, a new record has to beat the current SOTA by at least 0.005 nats with p < 0.01, run under 10 minutes on 8×H100s, evaluate under 10 minutes as well, and fit inside a decimal 16,000,000-byte artifact. Tokenizer changes are allowed, but the README says they’ll get extra scrutiny.  ￼

The frontier is much lower than the merged board. A community live tracker, updated Mar 26, 2026, 3:00 PM PT, lists the best pending “record-eligible” scores as #870 0.0935, #888 0.0942, #881 0.0990, #880 0.1003, and #868 0.1181. Those are all eval-time n-gram / full-rescore methods, not ordinary pure-neural improvements, and the tracker explicitly notes that the official leaderboard is lagging the frontier.  ￼

The PR map worth caring about is: #414 as the merged 11-layer base stack; #549 as the current official SOTA; #593 → #609 → #728 as the strongest pure-neural / quantization lineage; #659 → #846 → #870 / #888 as the n-gram full-rescore lineage; and #755 as the tokenizer wild card.  ￼

Most promising raw-score setup: build a #870 / #888-style full-rescore n-gram cache on top of a #728-strength neural base. That is my inference, but it is a strong one: #888 gets to 0.09420444 even though its own reference neural score is still 1.15945860, which means the win is mostly in the eval-time method, and the live tracker explicitly says a stronger neural base under the mixer should be able to go lower.  ￼

The caveat is legality/review risk. OpenAI reviewer valerio-oai said on #659 that the eval cache idea itself did not look illegal; the specific problem there was hindsight selection after seeing the true next token. But PR #886 argues that the deeper issue is whether these methods are violating the spirit of a 16 MB model by growing to hundreds of MB of eval-time state. So: not banned, but under review.  ￼

Most promising safe-ish official-track setup: start from #728’s self-generated full-GPTQ path, not from TTT. The stack I’d bet on is: 11L / 512d / 8H / 4KV, MLP 3×, LeakyReLU(0.5)^2, XSA on all 11 layers, BigramHash around 3072×112, Partial RoPE 16/64, layerwise LN scale, VE128 on late layers, SmearGate + U-Net skips, EMA 0.997, Parameter Banking + Parallel Muon, Full-Hessian GPTQ int6 + lzma, self-generated GPTQ calibration, and then selective ±1 pruning from #609. On Hopper, treat FlashAttention 3 as mandatory; #609 says FA2 costs enough step time to matter. This is basically #549 upgraded with the strongest pieces from #609 and #728.  ￼

I would not make TTT the centerpiece. #549 got about -0.0025 BPB from legal score-first TTT, which is real but modest. #728 says TTT was neutral or negative on the stronger XSA-all + full-GPTQ stack. #461 shows legal TTT can be very strong, but mostly as a lever on weaker bases, not the best current pure-neural line.  ￼

I would also not make tokenizer work your first mainline unless you are unusually strong on tokenization and metric verification. #755 is the biggest pure-model wild card I found—1.0321 BPB with a plain 12L 384d transformer and no XSA/EMA/TTT/sliding-window stack—but the repo rules explicitly say tokenizer edits will be examined much more carefully. Amazing upside, high review risk.  ￼

So my bottom line is:
	•	Highest raw upside: #728-grade neural base plus #870 / #888 full-rescore n-gram.
	•	Highest chance of an accepted official-style win: #728-style self-generated full-GPTQ neural stack, with #609 pruning, and TTT only as a late ablation.
	•	Wild-card moonshot: #755 tokenizer path, but only if you can prove BPB correctness cleanly.  ￼

Hard no’s: pre-eval TTT on unscored tokens, using training data after the 600s training budget for GPTQ calibration, or choosing between cache and LM after seeing the ground-truth token. Those are exactly the kinds of things the rules and reviewer comments are flagging.  ￼

My tactical order would be: reproduce #549 first, then swap in #728-style self-generated full GPTQ + XSA-all + bigger BigramHash + #609 pruning, and only then branch into #870 / #888 full-rescore n-gram. That gives you one line that can plausibly get merged and one line that can plausibly post the best absolute number.
