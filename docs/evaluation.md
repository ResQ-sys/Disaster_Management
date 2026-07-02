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

- Input: run without weather API key
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
