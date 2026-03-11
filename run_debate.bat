@echo off
cd /d "c:\Users\tamaghna roy\CascadeProjects\windsurf-project-3"
codex exec -m gpt-5.3-codex -c model_reasoning_effort=xhigh -s read-only --skip-git-repo-check --full-auto "Act as a quantitative finance researcher debating volatility model design. Read debate_round1.txt in the current directory and produce the full taxonomy requested. Output structured text with model families, targets, assumptions, complexity, and references."
