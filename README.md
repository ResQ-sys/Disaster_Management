# Disaster Relief Resource Matching Agent

AAI capstone project for disaster response coordination. The system accepts an incident report, retrieves relevant relief resources, ranks the best options, and generates a coordination-ready response for human review.

## Problem Statement

During disasters, affected people may need shelter, medical help, food, rescue support, transport, or evacuation. At the same time, local institutions and responders have limited available resources. This project builds an AI-assisted agent that:

- takes structured disaster input from a user
- enriches it with weather and district risk context
- matches the reported needs to available resources
- explains the top recommendation in simple English
- produces a coordination report for human approval

The goal is not only to run code, but to help a responder decide what action to take next.

## Real-World Impact

The project supports faster triage and coordination during emergencies while keeping a human decision-maker in control. It is designed for district control rooms, volunteers, campus teams, and local disaster coordinators who need a shortlist of actionable resource options.

## Dataset / Reference Sources

Primary reference sources used in this repository:

- Kaggle reference dataset: Disaster Tweets
  Link: https://www.kaggle.com/datasets/vstepanenko/disaster-tweets
  Use: contextual reference for disaster-related language and scenario framing
- Custom relief resource data in this project:
  - `data/needs.csv` — reported relief needs (volunteer worklist)
  - `data/resources.csv` — available volunteer/relief resources
  - `source_data/himachal_hospitals_289.csv`
  - `source_data/SCHOOL_GOVT_MARCH_2021.pdf`
  - `source_data/Shelter Info.pdf`
  - `source_data/TableViewStationForecastData.xlsx`
  - `source_data/Landslide Inventory Mapping (Post Monsoon for Himachal Pradesh) -2023.pdf`
  - `source_data/hp_glacial_lakes.csv` — GLOF risk, extracted from the CWC *Monthly Monitoring Report of Glacial Lakes, September 2025*
  - `source_data/Past_Data_For_Wildfire_detection_HP.csv` — VIIRS satellite fire-hotspot history (wildfire proneness)

Note: the custom `needs.csv` and `resources.csv` files are synthetic starter data, which is acceptable under the project brief when full operational data is not available.

## Tools Used

- Python
- Streamlit
- LangGraph
- LangChain
- Ollama (`llama3.2:1b`)
- ChromaDB
- sentence-transformers
- pandas
- folium
- Open-Meteo forecast API
- OpenRouteService API

## Project Workflow

The core response pipeline is a 6-node LangGraph agent:

1. User enters district, location, disaster type, and immediate needs (coordinates are auto-derived by geocoding).
2. **Intake agent** fetches weather, district risk, nearest CWC flood station, disaster knowledge, and a **wildfire-proneness** flag (from VIIRS fire history).
3. **GLOF monitor agent** flags expanding glacial lakes near the location (CWC Sept-2025 monitoring) — GLOF early warning.
4. **Resource finder** retrieves hospitals, shelters, and river-monitoring stations from ChromaDB.
5. **Matching agent** ranks the most suitable resources for the reported needs.
6. **Route planner** estimates travel distance/time and road risk to the resource.
7. **Escalation/report agent** computes an **explainable urgency score** and prepares a final report + coordination message.
8. A **human reviewer approves the coordination message** before any real action is taken.

Two additional interactive features sit alongside the agent:

- **Volunteer Need↔Resource Matching** — a deterministic engine matches `needs.csv` against `resources.csv` and drafts an approve-before-send coordination message per match.
- **Grounded RAG chatbot** — answers questions strictly from the ingested data, and refuses out-of-scope questions.

## AI / Agent Component

This project uses AI/logic meaningfully in several places:

- **Retrieval (RAG):** ChromaDB + sentence-transformers retrieve relevant hospitals, shelters, stations, glacial-lake, and disaster-knowledge records.
- **Matching:** an LLM ranks institutional resources; a separate **deterministic scoring engine** (`coordination.py`) matches volunteer needs↔resources by category, location, quantity, and availability (transparent, reproducible).
- **Urgency scoring:** an explainable 0–100 score combining IMD alert, district risk tier, need severity, GLOF, and wildfire signals — with a per-factor breakdown.
- **Spatial risk models:** wildfire proneness from historical fire-hotspot density; GLOF risk from glacial-lake area change.
- **Grounded RAG chatbot:** a relevance gate + strict grounding prompt prevent hallucination and answer only from source data.
- **Human-in-the-loop:** every coordination message is drafted for a human to edit → approve → send, logged to an audit trail (`logs/approvals.jsonl`). Nothing is dispatched autonomously.

Why this is useful:

- disaster inputs are noisy and incomplete
- multiple resource types may match the same need
- responders need an interpretable, auditable recommendation, not raw database output

## Repository Structure

```text
Disaster_Management/
├── app/
│   └── app.py
├── src/
│   └── resq_project/
│       ├── __init__.py
│       ├── config.py
│       ├── tools.py            # retrieval, weather, routing, wildfire, GLOF tools
│       ├── workflow.py         # 6-node LangGraph pipeline + urgency scoring
│       ├── chatbot.py          # grounded RAG chatbot (relevance gate)
│       └── coordination.py     # volunteer matching + human-in-loop messages
├── scripts/
│   ├── ingest.py                    # build ChromaDB collections
│   └── build_glacial_lakes_csv.py   # extract HP glacial lakes from CWC PDF
├── tests/
│   └── test_scenarios.py       # 18 deterministic validation tests
├── logs/
│   └── approvals.jsonl         # human-in-the-loop audit trail (created at runtime)
├── requirements.txt
├── data/
│   ├── needs.csv
│   └── resources.csv
├── docs/
│   ├── project_report.md
│   ├── presentation_outline.md
│   ├── evaluation.md
│   ├── responsible_use.md
│   └── demo_video_guide.md
└── source_data/
    └── reference files
```

This structure separates the Streamlit UI, backend logic, scripts, data, and documentation so the repository is easier to review and maintain.

## How to Run

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Make sure Ollama is running locally and the `llama3.2:1b` model is available:

```bash
ollama serve
ollama pull llama3.2:1b
```

3. Optional `.env` values:

```env
OLLAMA_BASE_URL=http://localhost:11434
ORS_API_KEY=your_key
AGENT_COORDINATOR_EMAIL=your_email@example.com
```

For the Streamlit demo, enter these directly in the sidebar:

- `Agent coordinator email`
- `User email`
- `SMTP password`

The app uses Gmail SMTP with hardcoded defaults (`smtp.gmail.com:587`, TLS).
Use a Gmail app password for the sender account.

4. Build the vector database:

```bash
python3 scripts/ingest.py --overwrite
```

5. Run the Streamlit app:

```bash
streamlit run app/app.py
```

6. (Optional) Run the validation test suite:

```bash
pytest -q          # 18 deterministic scenario tests
```

## Sample Inputs and Outputs

Example input:

- District: KULLU
- Disaster type: Flash Flood
- Needs: Medical, Shelter
- Location: Near Kullu bus stand

Expected output:

- district risk and weather summary
- prioritized relief resource
- route/risk note
- escalation flag if no suitable local resource is found
- emergency checklist and contact numbers

## Validation / Evaluation

Validation artifacts are included in `docs/evaluation.md`, plus an automated
suite in `tests/test_scenarios.py` (**18 passing** deterministic tests). The
project evaluates:

- whether a relevant resource is returned
- whether escalation is triggered when resources are missing
- whether the output contains an actionable next step
- whether the response remains understandable to a non-technical user
- urgency-score bands, volunteer-matching correctness, and the RAG relevance gate

Key measured results (see `docs/evaluation.md`):

- RAG gate: in-scope ≤0.59 vs out-of-scope ≥0.77 cosine distance → **0 false accepts/rejects** at threshold 0.70
- GLOF extraction: 12 increasing HP lakes — **exactly matches CWC's own reported figure**
- Volunteer matching: all 5 sample needs matched to the correct provider

The app also stores a stepwise `node_log` in state and a human-decision audit log (`logs/approvals.jsonl`), which serve as agent traces.

## Results and Insights

- The system produces structured, human-readable response reports instead of only returning raw matches.
- District risk and weather context improves prioritization quality.
- Retrieval plus LLM ranking gives more usable results than a static list lookup.

## Guardrails, Limitations, and Responsible Use

- This project is decision support, not an autonomous emergency dispatcher.
- Live weather comes from Open-Meteo and falls back to mock data only if the request fails.
- Route planning is approximate because exact resource coordinates are not always available.
- Real-time availability of hospitals, shelters, and volunteers is not guaranteed.
- All outputs must be verified by a human authority before action.

Detailed notes are in `docs/responsible_use.md`.

## Screenshots

Add prototype screenshots to the final submission package after running the app locally. The repo currently includes the app code and the documentation checklist, but not exported image files.

## Team Members

- Update with actual student names before final submission.

## Deliverables Status

- GitHub-ready repository: present
- Dataset/reference links and files: present
- Code files: present
- README: present
- Project report: present in `docs/project_report.md`
- Presentation outline: present in `docs/presentation_outline.md`
- Demo video guide/script: present in `docs/demo_video_guide.md`
- Requirements file: present
- Limitations and responsible-use notes: present

Still to be finalized outside this edit session:

- actual PPT or PDF slide deck export
- recorded 5 to 8 minute demo video
- real screenshots captured from a local run
- final team member names
