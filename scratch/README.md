# Scratch Directory Policy

This directory is designated strictly for ephemeral experiments, temporary prototyping, and exploratory debugging.

## Rules:
1. **No Production Code**: Active production components, models, or datasets must never reside in `/scratch`.
2. **Ephemeral Lifecycle**: Scratch files are not tracked for production deployments.
3. **Preservation**: If a scratch script contains valuable experimental findings, promote it to `research_archive/scripts/` or `src/`.
4. **Permanent Tests**: Diagnostic and regression tests must be promoted to `backend/test_chat_api.py` or `src/mlua/evaluation/`.
