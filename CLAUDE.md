# CLAUDE.md

This is a Python research repo. Prioritise correctness, reproducibility, and small reviewable changes.

## General rules

- Inspect relevant files before editing.
- Preserve existing experiment logic, seeds, splits, metrics, and output formats.
- Do not invent missing results, files, citations, or functions.
- Do not modify raw data or delete generated results unless explicitly told.
- Do not commit changes unless explicitly asked.

## Python style

- Prefer simple functions over classes unless a class is clearly appropriate.
- Prefer pure functions where practical.
- Organize files top-down: public entry points first, helpers below.
- Use descriptive names, including `is_`, `has_`, `can_`, or `should_` prefixes for booleans.
- Use named constants instead of magic numbers.
- Use type hints for public functions and complex internal functions.
- Use docstrings for complex functions, public APIs, and experiment entry points.
- Add comments only for non-obvious logic, assumptions, or “why”.
- Avoid broad exception handling unless adding useful context or re-raising.
- Avoid new dependencies unless necessary.

## Experiments

- Make parameters explicit.
- Validate config files, CLI arguments, loaded data, and assumptions before use.
- Avoid hard-coded absolute paths where possible.
- Log enough information to reproduce runs.
- Do not run expensive full experiments unless asked.
- For validation, prefer the smallest useful check.

## Tool use and permission rules

Before using any tool or command that may change the repository, system state, environment, or results, stop and ask for confirmation.

This includes:
- editing or deleting files
- creating new files, except short planning notes explicitly requested
- running training scripts or long experiments
- installing packages
- changing dependencies or environment files
- running commands that write outputs
- modifying git state
- committing, pushing, rebasing, or checking out branches
- accessing external services or APIs
- deleting, overwriting, or moving result files

For read-only inspection, Claude may proceed without asking. This includes:
- listing files
- reading source files
- reading LaTeX files
- reading reviews
- inspecting configs
- checking command help text
- running lightweight read-only commands such as `ls`, `find`, `grep`, `cat`, `head`, `sed -n`, and `git status`

When in doubt, ask first.

Before taking action, provide:
1. the intended command or tool use,
2. why it is needed,
3. what files or outputs it may affect.

## Writing style

- Use academic, scientific prose.
- Prefer clear, direct sentences over rhetorical or promotional language.
- Keep claims precise and proportional to the evidence.
- Avoid em dashes. Use commas, parentheses, or separate sentences instead.
- Avoid colons unless introducing a list, table, definition, or formal statement.
- Avoid semicolons unless they improve clarity.
- Avoid “AI-style” framing, including:
  - “It is important to note that”
  - “It should be noted that”
  - “This highlights”
  - “This underscores”
  - “This serves as”
  - “In conclusion”
  - “Furthermore” when “Additionally” or a direct sentence is clearer
  - “Delve into”, “robust”, “seamless”, “leverage”, “pivotal”, “crucial”, “significant” unless technically justified
- Avoid overclaiming. Do not use words such as “prove”, “guarantee”, “clearly”, or “obviously” unless the statement is formally established.
- Prefer active voice where it improves clarity.
- Prefer concrete subjects over vague subjects such as “this”, “it”, or “the approach” when ambiguity is possible.
- Do not add unnecessary transition sentences.
- Do not add broad motivational claims unless requested.
- Preserve the author’s intended meaning, notation, terminology, and level of caution.
- When revising manuscript text, make the smallest change needed to improve clarity, correctness, or flow.

## Paper / LaTeX

- Preserve notation, labels, references, and figure/table structure.
- Keep writing concise and academic.
- Do not change scientific claims unless supported by results or user instruction.
- If LaTeX fails, identify the first meaningful error.
- All manuscript edits must be wrapped in `\larry{new text}% Cx` so they render as coloured review text in the compiled PDF. Old text is preserved as `% Cx OLD: ...` comments above the new version. Structural changes that cannot be wrapped (e.g., `\subsection{}`) use a `% Cx: description` comment instead.
- Any `% Cx ...` tag must end the line. Unchanged text following on the same line is silently consumed by the LaTeX comment. Place the next sentence on a new line, and put required syntax (e.g., TikZ trailing `;`) before the `%`.
- Editing `main.tex` and `main.bib` does not require prior confirmation provided the `\larry{}` markup convention is followed and each change cites its C-item.

## Reviewer-response workflow

- When asked to address reviewer comments, first read `reviews/` and create or update `response_plan.md`.
- Do not treat `response_plan.md` as standing instructions. Treat it as a project artifact describing current revision tasks.
- Check `revisions.md` for the current edit status and markup convention before making any manuscript changes.
- Before editing code, running experiments, installing packages, or modifying manuscript files other than `main.tex` and `main.bib`, stop and ask for confirmation.

## Handover

After changes, summarise:
- files changed
- what changed
- what was tested
- risks or assumptions