# Frontend Code Quality Changes

## Summary

Added Prettier (formatting) and ESLint (linting) to the frontend development workflow, applied consistent formatting to all existing files, and created a root-level shell script for running quality checks.

---

## New Files

### `frontend/package.json`
Node.js project manifest. Declares dev dependencies and npm scripts:
- `npm run format` — auto-format all files with Prettier
- `npm run format:check` — check formatting without writing changes
- `npm run lint` — lint `script.js` with ESLint
- `npm run lint:fix` — auto-fix lint issues where possible
- `npm run quality` — run both `format:check` and `lint` in sequence

### `frontend/.prettierrc`
Prettier configuration:
- 4-space indentation (matches existing code style)
- Single quotes for JS/JSON
- 100-char print width (120 for HTML)
- Trailing commas in ES5 positions
- LF line endings

### `frontend/.prettierignore`
Excludes `node_modules/` and `package-lock.json` from Prettier formatting.

### `frontend/eslint.config.js`
ESLint flat config for `script.js` (ESLint v9 format). Key rules enforced:
- `no-var` — require `const`/`let`
- `prefer-const` — use `const` where variable is never reassigned
- `no-implicit-globals` — disallow implicit global declarations
- `curly` — always require curly braces on `if`/`else` blocks
- `eqeqeq` — require `===` over `==`
- `no-console` — warn on `console.log`; allow `console.error` and `console.warn`

### `scripts/check-frontend.sh`
Shell script runnable from the project root. Installs deps if needed, then runs Prettier check and ESLint. Accepts `--fix` flag to auto-fix issues instead of just reporting them.

```bash
# Check only
./scripts/check-frontend.sh

# Auto-fix
./scripts/check-frontend.sh --fix
```

---

## Changes to Existing Files

### `frontend/script.js`
1. **Wrapped all code in an IIFE** — eliminates implicit global function declarations; all state and helpers are now properly scoped.
2. **Added curly braces** to all single-line `if` bodies (3 locations).
3. **Removed debug `console.log` calls** — two development-only log statements removed; error logging with `console.error` retained.
4. **Applied Prettier formatting** — consistent quote style, trailing commas, line length.

### `frontend/style.css`
Applied Prettier formatting: consistent spacing around property values and selectors (no logic changes).

### `frontend/index.html`
Applied Prettier formatting: normalized attribute quoting and whitespace (no structure changes).
