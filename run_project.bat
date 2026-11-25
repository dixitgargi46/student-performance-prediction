@echo off
echo Installing required Python libraries...
python -m pip install -r requirements.txt
if errorlevel 1 goto error
echo Downloading dataset...
python download_data.py
if errorlevel 1 goto error
echo Creating EDA graphs...
python eda.py
if errorlevel 1 goto error
echo Training and comparing models...
python train_model.py
if errorlevel 1 goto error
echo Starting prediction application...
python predict.py
goto end
:error
echo Something went wrong. Read the error above and ask for help.
:end
pause
