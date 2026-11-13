# Demonstration Plan
## IoT Attack Diagnosis and Localization — Final Year Project

**Prerequisites before demo:**
- Virtual environment activated: `.venv\Scripts\Activate.ps1`
- All training scripts have been run (models exist in `models/`)
- Wireshark/TShark installed and on PATH

---

## Demo Step 1 — Show project structure (30 seconds)

```powershell
tree /F /A | Select-String -NotMatch "__pycache__|.venv|.git"
```

**Say:** "This is the project root. You can see the formal FSM models in `models_fsm/`, the ML training scripts in `ids/`, local MQTT captures in `captures/`, and the external 42-capture dataset in `data/base_paper/Packets/`."

---

## Demo Step 2 — Run unit tests (60 seconds)

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

**Expected output:** All 22+ tests pass.

**Say:** "All tests pass. The tests cover both EFSM state machines, the diagnosis rule helpers, dataset schema validation, model loading, and sequence generation math."

---

## Demo Step 3 — EFSM self-test: Temperature Sensor (45 seconds)

```powershell
.venv\Scripts\python.exe models_fsm\temperature_efsm.py
```

**Expected output:**
```
=== 1. Replay-like Detection ===
DETECTED: Replay-like: repeated payload  →  State: ANOMALOUS

=== 2. Data Falsification Detection ===
DETECTED: Data falsification: out-of-range temperature  →  State: ANOMALOUS

=== 3. Message Manipulation Detection ===
DETECTED: Message manipulation: payload changed  →  State: ANOMALOUS

=== 4. Excessive Communication Detection ===
DETECTED: Excessive communication: publish rate too high  →  State: ANOMALOUS
```

**Say:** "The EFSM has 7 states and 4 attack detectors. Each fires on a specific violation and transitions to ANOMALOUS."

---

## Demo Step 4 — EFSM self-test: Motion Sensor (30 seconds)

```powershell
.venv\Scripts\python.exe models_fsm\motion_efsm.py
```

**Say:** "The motion sensor EFSM handles binary payloads. Repeated values are evidence only, not an attack, because stable no-motion is legitimate."

---

## Demo Step 5 — Show rule-based diagnosis results (45 seconds)

```powershell
.venv\Scripts\python.exe ids\create_combined_diagnosis.py
```

**Key result:** 3 of 4 attack scenarios produce named signals. Replay produces evidence only.

---

## Demo Step 6 — Show RF baseline results (30 seconds)

```powershell
python -c "
import json; from pathlib import Path
d = json.loads(Path('results/base_paper_model_evaluation.json').read_text())
cr = d['classification_report']
print(f'Accuracy: {cr[\"accuracy\"]:.4f}  |  Benign F1: {cr[\"benign\"][\"f1-score\"]:.4f}  |  Attack F1: {cr[\"attack\"][\"f1-score\"]:.4f}')
"
```

**Say:** "96.1% accuracy, but benign F1 is only 0.58 due to class imbalance — one benign capture vs 41 attack captures."

---

## Demo Step 7 — Run unified diagnosis engine (45 seconds)

```powershell
.venv\Scripts\python.exe diagnosis_loc\unified_diagnosis.py
```

**Say:** "The unified engine combines ML predictions, EFSM signals, and rule evidence into a single structured record per capture. Each record explicitly states the localization level achieved."

---

## Demo Step 8 — Show a diagnosis record (30 seconds)

```powershell
python -c "
import json; from pathlib import Path
recs = json.loads(Path('results/unified_diagnosis_records.json').read_text())
rec = next(r for r in recs if 'battery' in r['capture_id'].lower())
for k,v in rec.items(): print(f'  {k:<35}: {v}')
"
```

---

## Demo Step 9 — Show model comparison (30 seconds, after training)

```powershell
.venv\Scripts\python.exe ids\compare_models.py
```

---

## Demo Step 10 — Show traceability matrix (30 seconds)

```powershell
python -c "
from pathlib import Path
lines = Path('docs/traceability_matrix.md').read_text().splitlines()
for l in lines:
    if any(x in l for x in ['✅','⚠️','❌','Status']): print(l)
"
```

**Say:** "Every component of the reference paper is mapped to its implementation status. We implemented 12 of 16 paper components. The missing ones — ultrasonic/RFID EFSMs and formal mutation score — are documented as out of scope."

---

## Viva Questions and Answers

| Question | Answer |
|---|---|
| Why is benign F1 only 0.58? | One benign capture available; benign test is a temporal slice of the same capture. Class imbalance (1:41 ratio) suppresses benign precision. |
| What does 100% capture-level accuracy mean? | 41 of 42 captures are attack; model consistency not generalisation. Explicitly disclaimed in evaluation JSON. |
| Why can't you localise the physical device? | IP = network endpoint, not physical device. NAT and IP spoofing make device identity unverifiable. Level 4 requires device certificates or physical topology data. |
| How is EFSM different from ML? | EFSM produces interpretable signals tied to state transitions. ML produces class probabilities. Together they provide both detection and explainability. |
| What is IAT and why does it matter? | Inter-arrival time between consecutive packets. Attack traffic often has regular or extremely low IAT (flooding) or irregular IAT (scanning). It's the paper's core temporal feature. |
