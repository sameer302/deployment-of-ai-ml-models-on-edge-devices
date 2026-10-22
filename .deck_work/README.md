# .deck_work: slide deck working copies (not in git)

This folder holds local working copies of the slide decks that Claude Code publishes to claude.ai.
Only this README is committed; everything else here is ignored by the root `.gitignore`.

The decks themselves are stored on claude.ai, so nothing is lost if this folder is deleted.
Claude reads each deck back from its link before editing it.

| Folder | Deck | Link |
|---|---|---|
| `memory_reduction/` | Edge Segmentation: Memory Reduction (step 0, experiment B, ...) | https://claude.ai/artifact/5PeBVMvAEEELyVFyLQTBdn |
| `literature/` | Edge Segmentation Literature Review (FlexNN, ...) | https://claude.ai/artifact/Ftr6DTxJtZDNgN44noCnkr |
| `experiments/` | Edge Segmentation Experiments (baseline, experiment A) | https://claude.ai/artifact/SBbJ3raXYSNNKHcEphRb8N |

A folder may hold only the slides that were edited; the full deck is always the one on claude.ai.

Each deck folder contains `project/deck.json` (slide order and sections) and
`project/slides/<id>.html` (one file per slide, with speaker notes in `<aside>`).
The links are private: share them from the deck's Share menu.
