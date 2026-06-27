# Vocab Learner

A Python CLI that turns scanned dictionary pages into structured vocabulary study sessions using vision LLMs.

![CI](https://github.com/vajapravin/vocab-learner/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Take a photo of a page from your physical dictionary. Get back a full study session: teaching cards for every word, plus a short story that weaves them together for reinforcement.

Built as a learning project in AI-powered application design — clean architecture, strict typing, provider-agnostic LLM abstraction, and reproducible dev environments via Dev Containers.

---

## Quickstart

**Requirements:** Docker Desktop, VS Code with the Dev Containers extension, and an API key from either [Anthropic](https://console.anthropic.com/settings/keys) or [OpenAI](https://platform.openai.com/api-keys).

```bash
# 1. Clone
git clone https://github.com/vajapravin/vocab-learner.git
cd vocab-learner

# 2. Set up your key
cp .env.example .env
# Edit .env and add your API key + set VOCAB_LEARNER_LLM_PROVIDER

# 3. Open in VS Code and reopen in container
code .
# Then: Cmd+Shift+P → "Dev Containers: Reopen in Container"

# 4. Once inside the container, run against your image
uv run vocab-learner path/to/dictionary_page.jpg
```

Output lands in `output/session_YYYYMMDD_HHMMSS.md` — one Markdown file per run.

A single page (~30 words) takes ~3 minutes and costs ~$0.10–0.30 depending on provider and model. Runs are printed to stdout as a path so you can pipe:

```bash
uv run vocab-learner page.jpg | xargs code
```

---

## What It Does

The system is a three-stage LLM pipeline behind a provider-agnostic client interface: