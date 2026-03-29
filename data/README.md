# Private Data Layout

This directory is intentionally ignored by git.

The public repository does **not** include the original Telegram export or media files. To rerun the experiments locally, place your private export here with the following structure:

```text
data/
  result.json
  photos/
  stickers/
  video_files/
```

The pipeline expects Telegram's standard export JSON where text messages are stored under the `messages` field and each row includes at least:

- `type`
- `date`
- `text`
- `from_id`

The repository never needs to publish these files. Keep them local and private.
