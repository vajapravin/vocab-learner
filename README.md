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
uv run vocab-learner run path/to/dictionary_page.jpg
```

Output lands in `output/session_YYYYMMDD_HHMMSS.md` — one Markdown file per run.

A single page (~30 words) takes ~3 minutes and costs ~$0.10–0.30 depending on provider and model. Runs are printed to stdout as a path so you can pipe:

```bash
uv run vocab-learner run page.jpg | xargs code
```

---

### Emailing a session

Once you have a generated `.md` file, email it as HTML:

    uv run vocab-learner send output/session_20260826_203242.md

Requires SMTP config in `.env` (see below). The email includes the rendered HTML in the body and the raw Markdown as an attachment.

## What It Does

The system is a three-stage LLM pipeline behind a provider-agnostic client interface:

## Example Input & Output

### Input: a photographed dictionary page

<a href="fixtures/page_012.jpg">
  <img src="fixtures/page_012.jpg" alt="A page from an English–Gujarati dictionary" width="400"/>
</a>

_A dense two-column page from an English–Gujarati dictionary — 30+ headwords, mixed Latin and Gujarati scripts, spine curvature, and normal photograph imperfections. Click to view full size._

### Output: a full study session

**→ [Read the full generated session](docs/example_session.md)** — 29 teaching cards + a reinforcement story, ~4,000 words of structured Markdown.

One card from the run, as a taste:

> ### 1. accession _(noun)_
>
> **Pronunciation:** ak-SESH-un · /əkˈsɛʃ.ən/
>
> **Meaning:** Accession means the act of joining or gaining a new position, often used when someone takes on an important role or when something is added officially.
>
> **Example:** _"After the CEO's accession to the company, major changes started happening."_
>
> **Where you'll see it:** You might see 'accession' in news articles about politics or business, especially when someone starts a new high-level job or when a country joins an organization.
>
> **Memory hook:** Imagine a king stepping up onto a throne; that's his accession – the moment he officially becomes king.
>
> **Word connections:**
> - **Synonyms:** entry, admission, attainment
> - **Antonyms:** resignation, removal
> - **Collocations:** accession to power, accession speech, accession to the throne
> - **Word family:** access, accessory, accessional
>
> **Try it yourself:** The accession of ___ to ___ changed everything.

### Running as a scheduled job

The `daily` subcommand is designed for use with an external scheduler (cron, systemd timer, Synology DSM Task Scheduler, etc.). It looks for `IMG_YYYY-MM-DD.jpg` in `VOCAB_LEARNER_INBOX_DIR` matching today's date, runs the pipeline, and emails the result.

    uv run vocab-learner daily                      # process today's file
    uv run vocab-learner daily --date 2026-08-27    # process a specific date
    uv run vocab-learner daily --strict             # exit with error if missing

The default (silent skip on missing file) is appropriate for scheduled runs. Use `--strict` for manual invocation when you want to be told if the file isn't there.