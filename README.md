# AEREAS

Academic Essay Review and Evaluation Agentic System

AEREAS is a backend platform for academic writing review. It evaluates submitted work, produces structured feedback, and can generate a revised draft based on detected issues. The system is designed for student and teacher workflows where writing quality, coherence, argument strength, tone, citations, and originality need to be assessed together.

## Overview

AEREAS combines rule-based analysis and optional model-assisted analysis in a single service. It processes either raw text or uploaded documents, then returns a consolidated review result that can include:

- Overall and category-level scores
- Actionable feedback items
- Quality control status
- Revised content output
- Change tracking metadata for review dashboards

The platform also supports persistence for review history and reporting so institutions can track writing progress over time.

## Core Features

- Multi-dimension writing assessment
  Grammar, coherence, argumentation, tone, citation quality, and plagiarism signals are evaluated in one run.

- Supervisor-driven orchestration
  A coordinator executes specialist checks, merges findings, and produces a unified outcome.

- Upload and extract pipeline
  Documents are accepted through object storage integration, text is extracted, and content is prepared for analysis.

- Review and revise mode
  In addition to feedback, AEREAS can produce a revised draft and summary of edits.

- Change tracking support
  Sentence-level before and after change data is generated for interfaces that need transparent revision views.

- Persistence and dashboard readiness
  Reviews, revisions, and document metadata can be stored for user-level and instructor-level history views.

- Structured logging and request correlation
  Centralized logs with correlation IDs improve traceability across upload, evaluation, and persistence paths.

## Workflow

1. Submission
   A user submits either text content or an uploaded file.

2. Ingestion and extraction
   The system validates file format, stores source data, and extracts readable text.

3. Evaluation
   Specialist analyzers review the content across multiple quality dimensions.

4. Synthesis
   Findings are consolidated into prioritized feedback and scoring output.

5. Optional revision
   If review and revise mode is enabled, a revised draft is produced and differences are tracked.

6. Persistence and response
   Results are returned to the caller and optionally saved for later dashboard and detail views.

## Project Scope

AEREAS is intended for:

- Academic writing support platforms
- Instructor feedback workflows
- Student self-review before submission
- Institutional writing quality analytics

## Running Locally

Install dependencies and start the API service in development mode.

```bash
uv sync
uvicorn main:app --reload
```

You can then use the interactive API documentation in your local environment to test upload, evaluation, and review flows.

## Configuration Notes

- Environment-based configuration is used for database, storage, and optional LLM settings.
- Model-backed analysis can be enabled or disabled depending on runtime capacity.
- Persistence-backed features require a configured database and applied migrations.

## Repository Layout

- app contains API routes, services, workers, schemas, and core runtime modules.
- main.py is the FastAPI application entry point.
- alembic and migration files manage schema evolution.
- docs contains design and deployment references.

## License

[MIT](LICENSE.txt)
