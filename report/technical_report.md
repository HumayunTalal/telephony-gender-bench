# Speaker gender detection on narrowband call audio

**Humayun Talal** · October 2026 · draft 0.2

Code: `github.com/HumayunTalal/telephony-gender-bench`

> Numbers in square brackets are placeholders. They are filled from `results/summary.csv` after a full run on the labelled set with the final front end. Nothing in this document is a number I have not measured.

## 1. The problem

The outbound voice agents I work on at Confinality speak to people over ordinary phone lines. For one Medicare campaign we wanted the agent to address the person as "sir" or "ma'am" from the second turn onward, and to switch back to neutral wording if it got that wrong. That needs a gender decision from the first one or two customer turns, within the turn gap, on 8 kHz audio.

The first version used a published gender model straight off the Hugging Face Hub, run on the customer turn audio with a light voice activity check in front of it. In production it was right about 80 percent of the time. That is not good enough to put "sir" in a script. A wrong "ma'am" costs more trust than a neutral line gains.

So the question became why a model that reports well above 90 percent on its own test set drops to 80 on ours, and what can be done about it without a large labelling effort.

## 2. What is different about this audio

Three things, and they stack.

**Bandwidth.** Telephony audio is sampled at 8 kHz, so nothing above 4 kHz survives. The published models are trained at 16 kHz or higher on read speech. Upsampling to 16 kHz before inference puts the signal in the right sample rate but does not put the missing band back. The model is seeing a version of its input with the top half of the spectrum empty.

**Length and content.** A first customer turn is often one word. "Hello?" "Yeah?" "Who is this?" There is very little voiced speech to work with, and what there is comes at the start of the clip, right where the line is still settling.

**Noise that looks like speech to a VAD.** Line hum, television, a dog, traffic, a second person in the room. The VAD we had in front of the model passed a good share of this as speech. The gender model was then asked to classify a clip with no customer voice in it, and it answered anyway, because it has no "noise" output. A confident wrong answer on a clip that should have been abstained from looks, in the logs, exactly like a confident wrong answer on a real voice.

## 3. Data

Customer-side turn audios taken from the first and second turns of calls that actually proceeded. Pickup-noise-and-disconnect calls were excluded because there is nothing in them to label.

Each clip was tagged by hand with two columns. `tag` is male, female or noise. `noise_flag` is set when there is audible background on top of the voice. The first pass used a single four-way label (male, female, noise, skip) and had to be redone, because it could not separate a voice with a TV behind it from a TV alone, and that is the distinction the whole problem turns on.

About 500 clips were tagged in the first pass, then more were added to reach roughly 2,000 in total. Doubtful and skipped clips were checked a second time. The tagging was done in a small Colab app that writes one CSV row per clip next to the clip list.

Class balance, clean versus noisy split and clip length distribution: [fill from `labels.summarise` and `results/*.csv`].

## 4. Method

### 4.1 Models

Two open models to start with, chosen because they are the two that were already in use or under test in production.

- `cv_gender`: prithivMLmods/Common-Voice-Gender-Detection, a wav2vec2 audio-classification head trained on Common Voice (read speech, 48 kHz source).
- `ecapa_gender`: JaesungHuh/voice-gender-classifier, an ECAPA-TDNN speaker encoder with a gender head trained on VoxCeleb (in-the-wild YouTube speech, 16 kHz).

The harness takes any Hugging Face audio-classification model with male/female labels, so more can be added with one line in the registry. [List any others that were run.]

### 4.2 Front ends

The same clip goes through each model three ways.

- `none`: raw 8 kHz clip upsampled to 16 kHz and peak-normalised. This is what production did.
- `energy`: a frame-energy VAD with an adaptive threshold and a hard floor, followed by a gate. If less than 0.6 s of speech or less than 15 percent of frames survive, the clip is marked noise and the model is not asked.
- `silero`: the same gate with Silero VAD as the detector.

The gate is deliberately strict in the direction production got wrong. Missing a bit of speech costs a little recall. Passing noise as speech costs a confident wrong label.

### 4.3 Scoring

Three numbers are always reported together, because any one of them alone can be made to look good.

- accuracy on all clips, where abstaining on a voiced clip counts as wrong
- accuracy on the clips the model actually answered
- coverage, the share of voiced clips it answered

Plus the share of pure-noise clips that were passed as a voice, which is the production failure mode by name, and per-class recall. Voiced clips are also split into clean and noisy so the cost of background noise is visible on its own.

### 4.4 Combining models at the call level

For tagging the call database, rather than a live decision, more than one model can be run and the votes pooled. Each call gets two labels. `label_1` is the vote over the first two customer turns. `label_2` is the vote over every customer turn. On a two-turn call they are the same. When they differ, the call goes on a QC list instead of getting a hard label. In practice the disagreements are where the agent's voice leaked into the customer channel on a later turn, or a second person took the phone.

A four-model version of this combiner was used to tag a 10,000-call database. A random sample of the tags was then checked by hand. [Fill in the QC rate and the sampled agreement.]

## 5. Results

### 5.1 Model by front end

[Paste the table from `scripts/report_tables.py results/`.]

| model | front end | n | acc (all) | acc (answered) | coverage | noise passed as voice | acc clean | acc noisy | ms/clip |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cv_gender | none | | | | | | | | |
| cv_gender | energy | | | | | | | | |
| cv_gender | silero | | | | | | | | |
| ecapa_gender | none | | | | | | | | |
| ecapa_gender | energy | | | | | | | | |
| ecapa_gender | silero | | | | | | | | |

### 5.2 What the confusion matrices say

[Fill. The two things to look for are the noise row under `none`, which should show how much of the production error was noise being answered, and the male/female cells on the noisy voiced clips, which show whether the models themselves are bad on this audio or only bad on the clips they should not have been given.]

### 5.3 Call-level tagging

[Fill from the 10k run. Share of calls with a hard label, share sent to QC, agreement on the hand-checked sample.]

## 6. What I take from this so far

Written before the full table is filled, so it is what the partial runs and the production logs already show, and it will be revised.

- A large part of the 80 percent was not the gender model being wrong about a voice. It was the model being asked about clips that had no customer voice in them. Fixing the gate moves the number before touching the model at all.
- The remaining errors concentrate on short, noisy clips, which is also where a human tagger slows down. Those are the clips to look at individually rather than to average over.
- Rules and preprocessing first, fine-tuning last. Fine-tuning needs labelled narrowband data, and the only realistic source of that is the tagged call database, which has to be QC'd before it can be trusted as training data. That is the next step, not this one.

## 7. Next

1. Full run of every model and front end on the ~2,000-clip set, fill section 5.
2. QC the 10k call tags using the disagreement list, then use the agreed calls as a fine-tuning set for a single model that can run live.
3. Measure the live constraint directly. The decision has to arrive inside the turn gap, so report latency per clip on the hardware the agent runs on, not just accuracy.
4. The open question I would like to spend real time on. How much of the loss on narrowband audio is recoverable by adapting the model to the band, versus inherent to the information that is left below 4 kHz. That is a question about representations, not about any one classifier.

## Appendix: reproducing

```bash
pip install -e ".[models]"
python scripts/benchmark.py --labels data/labels.csv --audio-dir clips/ --frontend none energy silero --out results/
python scripts/report_tables.py results/ > report/tables.md
```
