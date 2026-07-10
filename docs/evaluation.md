# Evaluation and Scenario Checks

## Objective

Validate that the project produces actionable, understandable, and safe outputs.

## Scenario 1: Medical Need in Kullu

- Input: Flash Flood, Kullu, needs = Medical and Shelter
- Expected:
  - weather and district risk shown
  - at least one hospital or shelter surfaced
  - one priority resource selected
  - response report includes next action

## Scenario 2: Shelter Need in Manali

- Input: Landslide, Kullu district, needs = Shelter
- Expected:
  - school or shelter option retrieved
  - road risk warning shown if applicable
  - emergency contacts included

## Scenario 3: No Local Match Found

- Input: district with limited resource coverage and unsupported need
- Expected:
  - `escalation_needed = True`
  - route planning skipped
  - final report asks for human escalation

## Scenario 4: API Fallback

- Input: run when the weather service is temporarily unavailable
- Expected:
  - mock/fallback weather data returned
  - workflow still completes
  - warning is visible in logs or report context

## Quality Checks

- Relevance: returned resource should fit the requested need category
- Actionability: report should say what to do next
- Transparency: output should mention warning/verification note
- Robustness: workflow should not stop when one API is unavailable

## Evidence in Code

- `src/resq_project/workflow.py`: agent stages, branching, escalation logic
- `src/resq_project/tools.py`: fallback logic and retrieval helpers
- `app/app.py`: user input and output explanation

## Suggested Metrics for Demo

- resource hit rate across sample cases
- escalation correctness for no-match cases
- completeness of final response sections
- user readability based on manual review

## Automated Test Suite

Deterministic checks live in `tests/test_scenarios.py`. Run with:

```bash
pytest -q
```

Result: **18 passed** (no LLM calls; pure logic + retrieval).

| Area | Test | What it validates |
|------|------|-------------------|
| Urgency scoring | high / low / monotonic | score bands (CRITICAL ≥75, LOW <30) and that worse inputs never lower the score |
| Volunteer matching | hard category filter | Medical need never matches a Food resource |
| Volunteer matching | location + coverage | nearer, better-stocked, verified provider scores higher |
| Volunteer matching | all sample needs matched | every needs.csv row finds a candidate |
| Volunteer matching | worklist ordering | queue sorted by need urgency |
| Coordination | message fields / escalation | draft contains request id + "approval required"; unmatched → escalate |
| Free-text extraction | tweet parsing | "URGENT: 30 trapped … rescue" → Rescue / Critical / qty 30 |
| RAG gate | in-scope accepted | HP hospital/GLOF/CWC questions clear the confidence threshold |
| RAG gate | out-of-scope rejected | France/cricket/cake all fall beyond threshold (refused before LLM) |
| Wildfire | prone vs remote | Shimla → PRONE (HIGH); Kaza-remote → not prone (0 within 10 km) |
| GLOF | increase flagged | Lahul & Spiti query returns ≥1 expanding lake |

## Measured Results (key numbers)

- **RAG relevance gate separation:** in-scope questions score cosine distance **≤0.59**, out-of-scope **≥0.77**; threshold set at **0.70** → **0 false accepts / 0 false rejects** on the test set.
- **GLOF extraction fidelity:** 17 HP glacial lakes extracted, **12 flagged increasing — exactly matching CWC's own "Himachal Pradesh-12" combined figure** in the Sept-2025 report.
- **Volunteer matching:** all 5 sample needs matched to the correct category provider (match scores 72–88/100, 100% coverage).
- **Wildfire calibration:** Shimla 233 / Dharamsala 130 / Manali 17 / Kaza 0 fire detections within 10 km → sensible HIGH/LOW/MINIMAL bands.
