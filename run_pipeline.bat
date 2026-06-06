@echo off
REM ============================================================
REM  run_pipeline.bat  —  Tourism Trend Analytics (BDT Project)
REM  Full pipeline execution with Apache Spark + Flask dashboard
REM  Run from the BDT_project root directory.
REM
REM  Execution Order:
REM    1. analytics.py              (Pandas: load, aggregate)
REM    2. sentiment.py              (TextBlob: sentiment scores)
REM    3. spark_analysis.py         (Apache Spark: big data analytics)
REM    4. hotspot.py                (Hotspot scoring formula)
REM    5. prediction.py             (Random Forest: leakage-free)
REM    6. processing_load_analysis  (Pandas vs Spark benchmark)
REM    7. google_trends.py          (OPTIONAL: real-time trends)
REM    8. app.py                    (Flask dashboard: localhost:5000)
REM ============================================================

setlocal EnableDelayedExpansion
set PIPELINE_START=%TIME%
set ERRORS=0

echo.
echo ============================================================
echo   TOURISM TREND ANALYTICS  -  FULL BDT PIPELINE
echo ============================================================
echo   Start time : %PIPELINE_START%
echo ============================================================
echo.

REM ─── Step 1: Analytics ───────────────────────────────────────
echo [STEP 1/7]  analytics.py  (load and aggregate dataset)
echo -----------------------------------------------------------
python scripts\analytics.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] analytics.py returned error %ERRORLEVEL%
    set /A ERRORS+=1
    pause
    exit /b 1
)
echo [PASS] analytics.py complete.
echo.

REM ─── Step 2: Sentiment ───────────────────────────────────────
echo [STEP 2/7]  sentiment.py  (TextBlob sentiment analysis)
echo -----------------------------------------------------------
python scripts\sentiment.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] sentiment.py returned error %ERRORLEVEL%
    set /A ERRORS+=1
    pause
    exit /b 1
)
echo [PASS] sentiment.py complete.
echo.

REM ─── Step 3: Spark Analysis ──────────────────────────────────
echo [STEP 3/7]  spark_analysis.py  (Apache Spark big data analytics)
echo -----------------------------------------------------------
echo   Demonstrating: SparkSession, DataFrame API, SQL, Window Functions,
echo   cache/persist, joins, groupBy aggregations, UDFs
python scripts\spark_analysis.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] spark_analysis.py returned error %ERRORLEVEL%
    echo   Check that PySpark is installed: pip install pyspark
    echo   Continuing pipeline...
    set /A ERRORS+=1
) else (
    echo [PASS] spark_analysis.py complete.
)
echo.

REM ─── Step 4: Hotspot Scoring ─────────────────────────────────
echo [STEP 4/7]  hotspot.py  (composite hotspot scoring)
echo -----------------------------------------------------------
python scripts\hotspot.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] hotspot.py returned error %ERRORLEVEL%
    set /A ERRORS+=1
    pause
    exit /b 1
)
echo [PASS] hotspot.py complete.
echo.

REM ─── Step 5: Prediction ──────────────────────────────────────
echo [STEP 5/7]  prediction.py  (leakage-free Random Forest model)
echo -----------------------------------------------------------
echo   Features: avg_rating, rating_std, avg_polarity, avg_subjectivity,
echo             city_avg_rating, city_place_count, city_avg_polarity,
echo             city_five_star_rate
echo   NOTE: city_review_count EXCLUDED to prevent target leakage
python scripts\prediction.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] prediction.py returned error %ERRORLEVEL%
    echo   Check that scikit-learn is installed: pip install scikit-learn
    set /A ERRORS+=1
) else (
    echo [PASS] prediction.py complete.
)
echo.

REM ─── Step 6: Processing Load Analysis ────────────────────────
echo [STEP 6/7]  processing_load_analysis.py  (Pandas vs Spark benchmark)
echo -----------------------------------------------------------
python scripts\processing_load_analysis.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] processing_load_analysis.py returned error %ERRORLEVEL%
    echo   Continuing pipeline...
    set /A ERRORS+=1
) else (
    echo [PASS] processing_load_analysis.py complete.
)
echo.

REM ─── Step 7: Google Trends (OPTIONAL) ────────────────────────
echo [STEP 7/7]  google_trends.py  (OPTIONAL: real-time Google Trends)
echo -----------------------------------------------------------
echo   NOTE: This step is optional. Failures are logged but do NOT
echo         stop the pipeline. Requires: pip install pytrends
python scripts\google_trends.py
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] google_trends.py did not succeed (this is OK).
    echo   The dashboard will show fallback trend scores.
) else (
    echo [PASS] google_trends.py complete.
)
echo.

REM ─── Pipeline Summary ────────────────────────────────────────
echo ============================================================
echo   PIPELINE SUMMARY
echo ============================================================
echo   Start time  : %PIPELINE_START%
echo   End time    : %TIME%
if %ERRORS% EQU 0 (
    echo   Status      : ALL STEPS PASSED
) else (
    echo   Status      : COMPLETED WITH %ERRORS% WARNING(S)
)
echo ============================================================
echo.

REM ─── Launch Flask Dashboard ──────────────────────────────────
echo [LAUNCH]  Starting Flask Dashboard ...
echo   URL: http://localhost:5000
echo   Press CTRL+C to stop the server.
echo.
python app.py

pause
