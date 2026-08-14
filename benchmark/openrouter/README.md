# Eval corpus (local only)

Image samples are **not** stored in git. Fetch them onto your machine when you want to run offline evals.

## Fetch

Needs `OPENROUTER_API_KEY` in `.env` for AI samples:

```bash
npm run fetch:openrouter   # AI gens into ai/<model-slug>/
npm run fetch:real         # Unsplash/Picsum reals into real/
```

`index.json` / `real-index.json` / `registry.json` are rewritten by those scripts.

## License note

Fetched images are for **local evaluation only**. Do not commit them — they are gitignored. Redistribution rights vary by model vendor and photo source.

## Without a corpus

Unit tests and CI use synthetic fixtures under `tests/fixtures/images/` (generated noise/gradients). `loadOpenRouterCorpus` falls back to those when this folder has no images on disk.
