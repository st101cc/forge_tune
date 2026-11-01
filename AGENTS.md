# Repository Guidelines

## Project Structure

The working application lives at the repository root:

- `index.html` is the app shell; `public/` holds the browser UI's `*.js` and `*.css` files.
- `server.mjs` provides the Node.js web server.
- `test/` holds the Node test scripts (`test-*.cjs`).
- `trainer/` contains the FastAPI/LangGraph Python service, training code, imports, evaluation, and Python tests.
- `validation/` and the Markdown reports hold small validation fixtures and experiment notes.
- `trainer/data/` and `.forge-data/` are runtime artifacts; do not commit generated databases, logs, models, or adapters.

## Build, Test, and Development

From the repository root:

```powershell
npm start
python -m pytest trainer -q
node test/test-datasets.cjs
```

The first command serves the UI at `http://127.0.0.1:4173`. Python tests cover the trainer and workflow logic. Browser-oriented scripts require the web server and Playwright; set `PLAYWRIGHT_MODULE` when the runtime does not provide it. For local GPU/VM startup, use `powershell -ExecutionPolicy Bypass -File start-auto.ps1`; `-CheckOnly` inspects the target and `-SelfTest` checks GPU selection.

## Coding Style and Naming

Use 2-space indentation in JavaScript and 4 spaces in Python. Keep the existing dependency-light, standard-library-first approach, native ES modules, and small functions. Use `camelCase` for JavaScript variables/functions, `snake_case` for Python names, `PascalCase` for Python classes, and descriptive kebab-free test names such as `test_dataset_analysis_pauses_without_gpu...`. Match nearby formatting; no formatter or linter is currently configured.

## Testing Guidelines

Test files and historical reports, specs and plans are git-ignored: they live in local checkouts only and are not published (see `.gitignore`). Add Python tests beside the code in `trainer/test_*.py`. Use pytest and temporary directories/mocks for filesystem, model, and workflow boundaries. Add Node tests as `test/test-*.cjs`. Run the narrowest relevant test first, then the full trainer suite before submitting.

## Commits and Pull Requests

History is minimal, so use concise imperative subjects, for example `Fix dataset split validation`. Keep commits focused. Pull requests should explain user-visible behavior, list verification commands, link an issue when one exists, and include screenshots for UI changes. Call out new environment variables, data migrations, GPU/VM assumptions, and any known limitations.

## Security and Configuration

Keep credentials in `trainer/.env` or the repository `.env`; commit only `.env.example` files. Never expose API keys to browser code or commit datasets, checkpoints, adapters, logs, or local runtime state.
