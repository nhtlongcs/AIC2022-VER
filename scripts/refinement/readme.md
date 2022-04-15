# 1. Gather SRL result

Input: srl_test_queries.json

Output: srl_postproc.csv (example in data/result/srl/postproc-2/srl_test_postproc.csv)

<!-- outdir at 'data/result/srl/postproc-2' -->

```bash
python external/srl/gather_result.py
```

# 2. Gather Video (Track) result

Gather relation/classification/action (stop-turn) results from different modules for each track to a single file.

**Input:**

- data/result/test_relation.json
- data/result/test_neighbors.json
- data/result/test_stop_turn.json

**Output:**

- save folder (example in data/result/test_relation)

<!-- Outdir at 'data/result/test_relation' -->

```bash
python external/refinement/parse_relation.py
```

# 3. Refinement

Input:

- srl.csv (output from 1st step)
- test_track_dir : output of step 2
- retrieval result: submission format

Output:

- final retrieval result: submission format

```bash
python external/refinement/main.py
```
