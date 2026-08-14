# Eval corpus

## AI samples (in git)

`ai/<model-slug>/` — one image per OpenRouter raster model, generated via the API for this project. Safe to keep in-repo.

Refresh incrementally (needs `OPENROUTER_API_KEY` in `.env`):

```bash
npm run fetch:openrouter   # only models not already in registry.json
```

## Real photos (local only)

`real/` is gitignored. Fetch Unsplash/Picsum samples when you want a balanced eval:

```bash
npm run fetch:real
```

## Not included

- Lexica scraped feeds / holdouts — not redistributed here
- Distill train corpora under `benchmark/distill-corpus/`

Without `real/` on disk, eval harnesses still run against the committed AI samples plus synthetic fixtures as needed.
