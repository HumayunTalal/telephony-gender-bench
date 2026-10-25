# telephony-gender-bench

Benchmark harness for speaker gender detection on narrowband (8 kHz) call-centre audio.

I built this because the gender classifier we ran in production on outbound call audio sat at roughly 80% accuracy, and nothing I read about the published models explained why. Most of them are trained and evaluated on 16 kHz or 48 kHz read speech (Common Voice, LibriSpeech, VoxCeleb). Our audio is 8 kHz telephony, a few seconds long, recorded in the first one or two turns of a cold call, with line noise, TV in the background, dogs, and people who say "hello?" and nothing else. The published accuracy numbers do not transfer.

This repo is the tooling I used to find out how far they fall and what helps. It does three things:

1. **Benchmark** several open gender models on a hand-labelled set of 8 kHz clips, with and without a preprocessing front end (voice activity detection, noise gating, loudness normalisation).
2. **Combine** the models at the call level (first two turns, and all turns) so that a whole call database can be tagged with a confidence flag and a disagreement list for manual QC.
3. **Report** accuracy, per-class recall, abstention rate and the confusion between male / female / noise, so the trade-off between coverage and accuracy is visible rather than hidden inside one number.

No audio and no labels are in the repo. The clips are customer call recordings and stay inside the company. The label schema, the folder layout and an example CSV are in `data/README.md` so the harness can be pointed at any similar collection.

## What is in here

```
tgb/                core package
  audio.py          load, resample 8k -> 16k, trim, peak-normalise
  vad.py            energy VAD (always available) and Silero VAD (optional), speech ratio, noise gate
  labels.py         two-column tag schema (male/female/noise + noise flag)
  metrics.py        accuracy, per-class recall, abstention-aware scores, confusion
  combine.py        per-clip vote across models, call-level merge (first two turns vs full call)
  models/           one wrapper per model family, all exposing the same predict() interface
scripts/
  benchmark.py      run every registered model over a labelled clip list, write results/
  tag_calls.py      tag a call database with the combiner, write labels + QC list
  report_tables.py  turn results/ into the markdown tables used in report/
tests/              unit tests for the parts that do not need a GPU or a model download
report/             technical report (markdown), figures go in report/figures/
data/               schema, layout, example label file (no audio)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[models]"      # torch, transformers, huggingface_hub
pip install -e ".[dev]"         # pytest
```

`pip install -e .` on its own gives you the audio, VAD, metrics and combiner code with only numpy, scipy, soundfile and pandas. The model wrappers import torch lazily, so the tests run without it.

Silero VAD is optional. If `torch` is installed it is loaded through `torch.hub` on first use; otherwise the energy VAD is used and the run log says so.

## Running the benchmark

Point it at a label CSV (see `data/README.md`) and a folder of clips:

```bash
python scripts/benchmark.py \
    --labels data/labels.csv \
    --audio-dir /path/to/clips \
    --models cv_gender ecapa_gender \
    --frontend none energy silero \
    --out results/
```

Each (model, frontend) pair writes a per-clip prediction CSV and one summary row. `frontend none` is the model on the raw 8 kHz clip upsampled to 16 kHz, which is what production did. The other front ends gate the clip through VAD first and abstain when too little speech is left.

```bash
python scripts/report_tables.py results/ > report/tables.md
```

## Tagging a call database

```bash
python scripts/tag_calls.py \
    --calls calls.csv \
    --audio-dir /path/to/turn_audio \
    --models cv_gender ecapa_gender \
    --out tagged/
```

`calls.csv` has one row per turn audio with a `call_id` and a `turn` index. The combiner produces two labels per call: `label_1` from the first two turns merged, and `label_2` from every turn merged. When they disagree the call goes on the QC list instead of getting a hard label. In practice the disagreements are where the agent's own voice leaked into the customer channel, or where the customer handed the phone to someone else.

## Label schema in one paragraph

Two columns per clip. `tag` is one of `male`, `female`, `noise`. `noise_flag` is either `noise` or empty. So a clean male voice is `male,` , a male voice with a TV behind it is `male,noise`, and a clip with no voice at all is `noise,noise`. This came out of an earlier single-column scheme (`male / female / noise / skip`) that could not separate "voice but noisy" from "no voice". The annotation was done in a small Colab app that writes the CSV next to the clip list.

## Status

- [x] energy VAD, Silero VAD wrapper, speech-ratio gate
- [x] wrappers for the two HF models we started with (`cv_gender`, `ecapa_gender`)
- [x] combiner with label_1 / label_2 and QC list
- [x] metrics with abstention accounting
- [ ] wav2vec2 fine-tune on the tagged set (needs the 10k-call tags to be QC'd first)
- [ ] report: fill the results tables from a full run with the final front end

See `report/technical_report.md` for the write-up and `CHANGELOG.md` for what changed when.

## Licence

MIT. The models themselves carry their own licences on the Hugging Face Hub.
