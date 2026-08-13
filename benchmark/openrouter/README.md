# OpenRouter eval corpus

Stored AI samples from every OpenRouter raster image-generation model, plus real photos for balance.

## Layout

- `registry.json` — which model IDs have already been fetched (`ok` / `skipped` / `error`)
- `index.json` — AI image inventory
- `ai/<model-slug>/` — one sample image per model
- `real/` + `real-index.json` — real photographs (Lorem Picsum / Unsplash)

## Refresh (incremental)

Requires `OPENROUTER_API_KEY` in the environment or `.env` (gitignored).

```bash
npm run fetch:openrouter   # only models not already in registry.json
npm run fetch:real         # only real photo IDs not already downloaded
```

Vector Recraft models are skipped on purpose (not useful for photo-detector eval) and recorded as `skipped` so they are not retried.

Force regeneration of everything:

```bash
OPENROUTER_FORCE=1 npm run fetch:openrouter
```

## Notes

- One prompt per model keeps cost bounded while covering the full catalog.
- Re-running `fetch:openrouter` after new models appear on OpenRouter only downloads the newcomers.
