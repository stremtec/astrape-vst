## Summary

<!-- What does this PR do? One paragraph. -->

## Motivation

<!-- Why is this change needed? Link related issues. -->

## Type

- [ ] feat — new feature
- [ ] fix — bug fix
- [ ] refactor — code restructuring (no behavior change)
- [ ] perf — performance improvement
- [ ] docs — documentation only
- [ ] ci — CI/CD changes
- [ ] test — test additions/fixes
- [ ] chore — maintenance (deps, gitignore, etc.)

## Checklist

- [ ] Streaming invariance holds — `python tests/test_streaming_invariant.py` passes
- [ ] All tests pass — `pytest tests/ -v`
- [ ] Lint clean — `ruff check astrape/ tests/`
- [ ] Type check clean — `mypy astrape/ --ignore-missing-imports`
- [ ] Imports are intact — all `astrape.*` modules import without error
- [ ] No look-ahead introduced — strictly causal (0 future frames)
- [ ] Architecture doc updated if latency/capacity changed

## Verification

<!-- How did you verify this works? Commands run, output checks, etc. -->
