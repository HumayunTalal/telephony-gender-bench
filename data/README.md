# Data layout

Nothing in this folder is audio. The clips are customer call recordings and do not leave the company. What is here is the schema, so the harness can be pointed at any similar collection.

## Clips

Short customer-side turn audios, 8 kHz mono, a few seconds each. We took them from the first and second customer turns of calls that actually proceeded (turn 1 plus a turn 2 or 3), and skipped the pickup-noise-and-disconnect calls, because those have no voice to label.

Put them in one folder. File names are whatever the dialler produced; the label CSV refers to them by name.

## Label file (`labels.csv`)

```
clip,tag,noise_flag
20251003_114522_3041.wav,male,
20251003_114530_3042.wav,female,noise
20251003_114601_3045.wav,noise,noise
```

| column | values | meaning |
|---|---|---|
| `clip` | file name | relative to `--audio-dir` |
| `tag` | `male`, `female`, `noise` | what the main voice is, or `noise` if there is no voice |
| `noise_flag` | `noise` or empty | is there audible background on top of the voice |

So:

- clean voice: `male,` or `female,`
- voice with background: `male,noise` or `female,noise`
- no voice at all: `noise,noise`

The gold label a model is scored against is just `tag`. `noise_flag` splits the voiced clips into clean and noisy so the summary can show accuracy on each.

### Why two columns

The first pass used a single column, `male / female / noise / skip`. That could not tell "a woman with a TV on" from "a TV and nothing else", and the whole point of the study was that the production model was failing on exactly that distinction. The second pass re-tagged with two columns. `skip` went away; a clip you cannot label is `noise,noise`.

### How the tagging was done

A small Colab app that plays each clip and writes the row (serial number, link to the turn audio, the two tags) to a CSV in the same Drive folder as the input list. One person tagged the first ~500 clips, then more were tagged to reach about 2,000 in total. The skipped and doubtful clips were checked a second time.

## Call file (`calls.csv`) for `tag_calls.py`

```
call_id,turn,clip
A1B2C3,1,A1B2C3_t1.wav
A1B2C3,2,A1B2C3_t2.wav
A1B2C3,3,A1B2C3_t3.wav
```

One row per customer turn, `turn` 1-based. The combiner makes `label_1` from turns 1 and 2 and `label_2` from all turns.

## Example

`labels.example.csv` has a handful of rows in the right shape. It points at files that do not exist; it is only there so `tgb.labels.read_labels` has something to validate against in the tests.
