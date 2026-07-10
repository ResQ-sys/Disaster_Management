# Project Report

## Title

Disaster Relief Resource Matching Agent

## Domain and Track

- Track: AAI
- Domain: Disaster response coordination
- Difficulty: Basic + Intermediate

## Problem Understanding

In a disaster, affected people report urgent needs such as shelter, medical support, food, transport, and water. Responders must quickly identify which resources are relevant, safe, and worth escalating to authorities. Manual triage is slow, inconsistent, and difficult under pressure.

This project addresses that gap with a human-supervised AI assistant that organizes inputs, retrieves likely resources, prioritizes them, and produces a report that tells the responder what to do next.

## Users and Stakeholders

- affected citizens or volunteer operators entering the case
- district control rooms reviewing recommended actions
- NGOs or student teams coordinating local support
- human authorities who approve final outreach and escalation

## Inputs

- district
- location description
- coordinates
- disaster type
- immediate needs
- supporting reference data and resource records

## Outputs

- structured disaster report
- ranked resource match
- escalation recommendation
- route/risk note
- emergency contact checklist
- agent step log

## System Overview

The system is implemented as a LangGraph workflow with six stages:

1. intake agent
2. GLOF monitor agent
3. resource finder agent
4. matching agent
5. route planning agent
6. escalation/report agent

This design was chosen because the workflow has clear branching. If no relevant resource is found, the system skips route planning and escalates directly.

## Data Preparation

The project uses a mix of public reference material and synthetic starter data:

- Himachal Pradesh hospitals CSV
- school PDF used as shelter proxy
- shelter information PDF
- CWC station Excel file
- landslide risk PDF
- synthetic `needs.csv`
- synthetic `resources.csv`

The synthetic CSVs are included because operational, real-time relief data is difficult to obtain. This is explicitly allowed by the capstone brief.

## AI / ML / Agent Logic

- Retrieval: sentence-transformers embeddings with ChromaDB collections
- Agent workflow: LangGraph state machine
- Prioritization: LLM-based ranking and explanation
- Guardrails: escalation path, warning note, and human verification note

## Why AI Is Useful Here

- the same need can map to multiple resource types
- context such as flood risk and weather changes the best choice
- responders need a simple explanation, not raw records
- escalation needs to be structured for quick review

## Validation Approach

The project is validated with scenario-based checks in `docs/evaluation.md`.

Questions used for validation:

- Was a relevant resource suggested?
- Did the agent escalate when no resource was available?
- Did the report tell the user what to do next?
- Were limitations and warnings clearly included?

## Responsible Use

- no automatic dispatch or contact is performed
- output must be checked by a human
- incomplete data can reduce matching quality
- the system should not be used as a substitute for official emergency command systems

## Conclusion

The project satisfies the core capstone goal by providing a usable, explainable, and action-oriented disaster resource matching workflow with a clear AI component and human oversight.
