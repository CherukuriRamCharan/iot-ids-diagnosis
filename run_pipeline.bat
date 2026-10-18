@echo off
:: ============================================================
:: run_pipeline.bat — Full IoT IDS pipeline runner
:: Run from project root: run_pipeline.bat
:: ============================================================

setlocal
set ROOT=%~dp0
set PY=%ROOT%.venv\Scripts\python.exe

echo ============================================================
echo  IoT Attack Diagnosis — Full Pipeline
echo  Project: %ROOT%
echo ============================================================
echo.

:: ---- Step 1: Tests ----
echo [STEP 1] Running unit tests...
%PY% -m pytest tests\ -v 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Some tests failed. Check output above.
    echo          Continuing with pipeline...
)
echo.

:: ---- Step 2: EFSM self-tests ----
echo [STEP 2] Running EFSM self-tests...
%PY% models_fsm\temperature_efsm.py 2>&1
echo ---
%PY% models_fsm\motion_efsm.py 2>&1
echo.

:: ---- Step 3: IAT feature generation ----
echo [STEP 3] Generating IAT features (this takes 20-30 min)...
if exist "%ROOT%data\base_paper\all_packet_features_iat.csv" (
    echo [SKIP] all_packet_features_iat.csv already exists. Delete it to regenerate.
) else (
    %PY% data\base_paper\add_iat_features.py 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] IAT generation failed! Check errors above.
        pause
        exit /b 1
    )
)
echo.

:: ---- Step 4: 5-class RF ----
echo [STEP 4] Training 5-class Random Forest...
%PY% ids\train_rf_5class.py 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] RF 5-class training failed!
    pause
    exit /b 1
)
echo.

:: ---- Step 5: LSTM ----
echo [STEP 5] Training LSTM (this takes 15-30 min on CPU)...
%PY% ids\train_lstm.py 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] LSTM training failed!
    pause
    exit /b 1
)
echo.

:: ---- Step 6: Transformer ----
echo [STEP 6] Training Transformer (this takes 15-30 min on CPU)...
%PY% ids\train_transformer.py 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Transformer training failed!
    pause
    exit /b 1
)
echo.

:: ---- Step 7: Predict on all 42 captures (binary RF) ----
echo [STEP 7] Running predictions on all 42 captures...
%PY% ids\predict_capture_attacks.py 2>&1
echo.

:: ---- Step 8: Combined ML + rule diagnosis ----
echo [STEP 8] Running combined diagnosis on all captures...
%PY% ids\run_combined_diagnosis.py 2>&1
echo.

:: ---- Step 9: Capture-level evaluation ----
echo [STEP 9] Running capture-level evaluation...
%PY% ids\final_evaluation.py 2>&1
echo.

:: ---- Step 10: MQTT rule diagnosis ----
echo [STEP 10] Running MQTT payload rule evaluation...
%PY% ids\evaluate_rules_on_captures.py 2>&1
%PY% ids\evaluate_packet_rates.py 2>&1
%PY% ids\create_combined_diagnosis.py 2>&1
echo.

echo ============================================================
echo  Pipeline COMPLETE
echo  Check results\ folder for all output files:
echo    results\rf_5class_evaluation.json
echo    results\lstm_evaluation.json
echo    results\transformer_evaluation.json
echo    results\combined_attack_diagnosis.csv
echo    results\combined_diagnosis_report.csv
echo    results\final_capture_evaluation.json
echo ============================================================
echo.
pause
