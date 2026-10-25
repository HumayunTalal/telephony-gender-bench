# Changelog

## 0.2.0

- Two-column label schema (`tag` + `noise_flag`) replaces the old four-way label. `read_labels` validates it.
- Energy VAD now adapts its threshold to the clip's noise floor, with a hard floor so hum does not count as speech. Fixed the case where a clip that is speech end to end got zero speech frames.
- VAD runs on the clip at its recorded level; peak normalisation happens after the gate, not before (before, a hiss-only clip got lifted to full scale and passed as speech).
- Gate abstains below 0.6 s of speech or 15% speech frames; abstentions are scored as wrong on voiced clips.
- Call-level combiner with `label_1` (first two turns) and `label_2` (all turns), QC list on disagreement, tie or low confidence.
- `benchmark.py` writes a clean/noisy split for voiced clips.
- Tests for VAD, gate, combiner, labels, metrics, and the pipeline with a fake model.

## 0.1.0

- First pass: load 8k clips, upsample, run `cv_gender` and `ecapa_gender` with no front end, write a per-clip CSV. This is roughly what production was doing.
