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
  - `data/needs.csv`
  - `data/resources.csv`
  - `source_data/himachal_hospitals_289.csv`
  - `source_data/SCHOOL_GOVT_MARCH_2021.pdf`
  - `source_data/Shelter Info.pdf`
  - `source_data/TableViewStationForecastData.xlsx`
  - `source_data/Landslide Inventory Mapping (Post Monsoon for Himachal Pradesh) -2023.pdf`

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

1. User enters district, location, disaster type, coordinates, and immediate needs.
2. Intake agent fetches weather, district risk, nearby flood station context, and disaster knowledge.
3. Resource finder retrieves hospitals, shelters, and monitoring resources.
4. Matching agent ranks the most suitable resources based on need and context.
5. Route planner estimates travel route risk when a priority resource is available.
6. Escalation/report agent prepares a final response report and emergency coordination message.
7. Human reviewer verifies the recommendation before any real action is taken.

## AI / Agent Component

This project uses AI meaningfully in three places:

- Retrieval: ChromaDB plus sentence-transformers is used to retrieve relevant hospitals, shelters, stations, and disaster knowledge.
- Matching: an LLM ranks candidate resources and explains why the top option was selected.
- Human-in-the-loop reporting: the agent generates a structured coordination note rather than acting autonomously.

Why this is useful:

- disaster inputs are noisy and incomplete
- multiple resource types may match the same need
- responders need an interpretable recommendation, not raw database output

## Repository Structure

```text
Disaster_Management/
├── app/
│   └── app.py
├── src/
│   └── resq_project/
│       ├── __init__.py
│       ├── config.py
│       ├── tools.py
│       └── workflow.py
├── scripts/
│   └── ingest.py
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
```

4. Build the vector database:

```bash
python3 scripts/ingest.py --overwrite
```

5. Run the Streamlit app:

```bash
streamlit run app/app.py
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

Validation artifacts are included in `docs/evaluation.md`. The project evaluates:

- whether a relevant resource is returned
- whether escalation is triggered when resources are missing
- whether the output contains an actionable next step
- whether the response remains understandable to a non-technical user

The app also stores a stepwise `node_log` in state, which serves as an agent trace.

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
