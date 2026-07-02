# Responsible Use Notes

## Intended Use

This project is an AI-assisted coordination helper for educational and prototype purposes. It supports human responders by organizing information and generating recommendations.

## Not Intended For

- autonomous emergency dispatch
- unverified medical advice
- replacing district administration or official command systems

## Guardrails Included

- escalation path when no local resource is found
- explicit verification note in the report
- human-in-the-loop final decision
- fallback behavior when APIs fail

## Risks

- stale resource records
- incomplete availability information
- inaccurate coordinates or user input
- approximate routing due to missing exact facility geocodes

## Mitigations

- verify all output before acting
- maintain updated local resource data
- log scenario runs and review bad cases
- use official helplines and control rooms for confirmation
