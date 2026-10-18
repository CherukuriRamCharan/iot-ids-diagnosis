# ============================================================
# run_pipeline.ps1 — Full IoT IDS pipeline runner (PowerShell)
# Run from project root:  .\run_pipeline.ps1
# ============================================================

$ErrorActionPreference = "Continue"
$ROOT  = Split-Path -Parent $MyInvocation.MyCommand.Path
$PY    = Join-Path $ROOT ".venv\Scripts\python.exe"

function Step($n, $msg) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " [STEP $n] $msg" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function RunPy($script) {
    $fullPath = Join-Path $ROOT $script
    Write-Host "Running: $fullPath" -ForegroundColor Yellow
    & $PY $fullPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAILED] Exit code: $LASTEXITCODE" -ForegroundColor Red
    } else {
        Write-Host "[OK]" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "IoT Attack Diagnosis — Full Pipeline" -ForegroundColor White
Write-Host "Project root: $ROOT"
Write-Host "Python:       $PY"
Write-Host ""

# ---- Step 1: Tests ----
Step 1 "Unit tests"
& $PY -m pytest (Join-Path $ROOT "tests\") -v
Write-Host ""

# ---- Step 2: EFSM self-tests ----
Step 2 "EFSM self-tests"
RunPy "models_fsm\temperature_efsm.py"
RunPy "models_fsm\motion_efsm.py"

# ---- Step 3: IAT features ----
Step 3 "Generate IAT features (~20-30 min)"
$iatCsv = Join-Path $ROOT "data\base_paper\all_packet_features_iat.csv"
if (Test-Path $iatCsv) {
    Write-Host "[SKIP] $iatCsv already exists." -ForegroundColor Yellow
    Write-Host "       Delete it and rerun to regenerate."
} else {
    RunPy "data\base_paper\add_iat_features.py"
}

# ---- Step 4: 5-class RF ----
Step 4 "Train 5-class Random Forest"
RunPy "ids\train_rf_5class.py"

# ---- Step 5: LSTM ----
Step 5 "Train LSTM (~15-30 min on CPU)"
RunPy "ids\train_lstm.py"

# ---- Step 6: Transformer ----
Step 6 "Train Transformer (~15-30 min on CPU)"
RunPy "ids\train_transformer.py"

# ---- Step 7: Predict on 42 captures ----
Step 7 "Predict on all 42 captures (binary RF)"
RunPy "ids\predict_capture_attacks.py"

# ---- Step 8: Combined ML + rule diagnosis ----
Step 8 "Combined ML + rule diagnosis"
RunPy "ids\run_combined_diagnosis.py"

# ---- Step 9: Capture-level evaluation ----
Step 9 "Capture-level final evaluation"
RunPy "ids\final_evaluation.py"

# ---- Step 10: MQTT rule diagnosis ----
Step 10 "MQTT payload + rate rule diagnosis"
RunPy "ids\evaluate_rules_on_captures.py"
RunPy "ids\evaluate_packet_rates.py"
RunPy "ids\create_combined_diagnosis.py"

# ---- Step 11: Model comparison ----
Step 11 "Compare all model results"
RunPy "ids\compare_models.py"

# ---- Step 12: Unified diagnosis engine ----
Step 12 "Run unified diagnosis and localization engine"
RunPy "diagnosis_loc\unified_diagnosis.py"

# ---- Summary ----
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Pipeline COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Output files in results\:"
$resultFiles = @(
    "rf_5class_evaluation.json",
    "lstm_evaluation.json",
    "transformer_evaluation.json",
    "model_comparison.json",
    "unified_diagnosis_records.json",
    "unified_diagnosis_records.csv",
    "combined_attack_diagnosis.csv",
    "combined_diagnosis_report.csv",
    "final_capture_evaluation.json",
    "base_paper_model_evaluation.json"
)
foreach ($f in $resultFiles) {
    $p = Join-Path $ROOT "results\$f"
    if (Test-Path $p) {
        $size = [math]::Round((Get-Item $p).Length / 1KB, 1)
        Write-Host "  [OK] $f  ($size KB)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $f" -ForegroundColor Red
    }
}
Write-Host ""
