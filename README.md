# AgriSense AI — Smart Agriculture & Irrigation Planning

A local Flask + scikit-learn educational project with a farmer-facing dashboard and an AIML Experiment Studio. No login is required for the mini-project demonstration.

## Run in VS Code on Windows
1. Extract this ZIP into a new folder.
2. In VS Code choose **File → Open Folder** and select the extracted `AgriSense_Professional_Website` folder.
3. Open **Terminal → New Terminal** and run one command at a time:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-cache-dir -r requirements.txt
python app.py
```

4. Open `http://127.0.0.1:5000` in a browser. Keep the terminal open while using the website. Stop the server with Ctrl+C. If you already have a working `.venv`, activate it and run the install command rather than creating another environment.

## Implemented modules
- **Farm dashboard:** supplied sensor sample, summary cards and dataset charts.
- **Irrigation advisor:** real scikit-learn pipelines for KNN, Decision Tree, Logistic Regression, Naive Bayes and Random Forest; probability and transparent rule-based caveats.
- **Model evaluation:** accuracy, precision, recall, F1, baseline, stratified cross-validation and model comparison.
- **Unsupervised learning:** K-means, agglomerative clustering, DBSCAN and PCA projection.
- **Regression:** Linear Regression with MAE, RMSE, R², coefficients and actual-vs-predicted plot.
- **ANN:** MLPClassifier with ReLU hidden layers and Adam optimizer.
- **Semi-supervised learning:** SelfTrainingClassifier pseudo-labeling demonstration.
- **Frequent-pattern mining:** Apriori-style itemset counting with adjustable support on discretized field records.
- **Search:** A* grid route, BFS/DFS/UCS graph traversal, Water Jug BFS, AO* AND-OR graph evaluation and Minimax with alpha-beta pruning.
- **Optimization:** Hill Climbing and Simulated Annealing irrigation-schedule comparison; TSP nearest-neighbour baseline vs simulated-annealing tour.
- **CSP:** backtracking assignment of irrigation time slots subject to adjacency and capacity constraints.
- **Reinforcement learning:** toy Q-learning environment with a learned wait/irrigate policy.
- **Image processing basics:** upload an image to inspect dimensions, mean RGB, brightness and a simple green-pixel ratio heuristic. This is not crop-disease detection or YOLO.
- **Dataset profiling:** missing values, numeric summary and target balance.
- **Experiment tracking:** save manually recorded experiment runs to `data/experiment_runs.csv`. This is a lightweight local CSV log, not MLflow.
- **Exports:** download the synthetic training CSV.

## Project structure
- `app.py` — Flask routes, model training and algorithm endpoints.
- `templates/index.html` — website and interactive experiment panels.
- `static/style.css` — responsive layout and styles.
- `data/synthetic_training_data.csv` — 1,200 synthetic rows for reproducible algorithm demos.
- `data/sensor_data.csv` — supplied small sensor sample.
- `data/nasa_power_raw.csv` — supplied NASA POWER export (single date).
- `REPORT_GUIDE.md` — report writing guidance and honest scope notes.

## Important scientific limitations
The training dataset is synthetic. The `irrigation_needed` target is created from illustrative proxy rules, not independently verified farm outcomes. Therefore strong model metrics mean the model learned this synthetic labeling pattern; they do **not** establish real-world irrigation accuracy or water savings. The 43-row sensor sample is not a robust training dataset. NASA POWER export covers only one date and is not currently integrated into training.

Optimization, CSP, RL, AO*, Minimax, Water Jug and TSP use small illustrative environments and are included for AIML coursework/viva demonstrations, not real pump control. Image analysis is a simple pixel heuristic. OpenCV-based advanced processing, trained YOLO object detection, live weather APIs, physical sensors, hardware actuation, cloud deployment, full AutoML and MLflow are **not** implemented in this version. Do not claim these as completed features.

The website loads Chart.js and Google Fonts through CDNs, so charts/fonts need internet access. Python backend APIs run locally.

## Farmer-first multilingual redesign
The interface uses a hybrid product layout: the primary navigation is designed around farm tasks (Overview, My farm, Water management, Reports & data), while model metrics and syllabus experiments are grouped under the Student / Developer area. Use the language selector in the top-right to switch the main interface between English, Hindi and Marathi. Some detailed algorithm output text remains technical/English because it is generated dynamically for the viva and diagnostics.

## Important scope notes
- The current training dataset is synthetic, and irrigation labels are generated for demonstration. Do not use model outputs as field-validated agricultural advice.
- The supplied sensor sample is small. No live weather service, physical sensor, pump control, or trained YOLO crop-disease model is connected.
- Basic image analysis reports RGB/green-pixel statistics only; it does not diagnose plant disease.
- Run the experiments locally and verify each output before a live demonstration. The model diagnostics area is intended for technical explanation and viva, not for farmer-facing decisions.


## Account sign-in

- Open the app and choose **Create an account** on the sign-in page.
- Registration is stored in `data/agrisense_users.sqlite3`; passwords are stored as salted hashes, not plain text.
- After signing in, the farmer dashboard and API routes require an authenticated session. Use **Sign out** to end the session.
- This is a local educational prototype. Before public deployment, configure a strong `AGRISENSE_SECRET_KEY`, add CSRF protection, HTTPS, rate limiting, email verification and a production server.


### First-time Windows setup

From the project folder in VS Code terminal (Command Prompt):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --no-cache-dir -r requirements.txt
python app.py
```

Alternatively, double-click `START_APP.bat`; it creates `.venv` if needed, installs requirements, and launches Flask. When the terminal says `Running on http://127.0.0.1:5000`, open that address in your browser. Choose **Create an account** the first time you visit.

### Re-evaluating model performance
On the **Model performance & validation** page, adjust the test split, KNN neighbours, Decision Tree depth, or Random Forest tree count, then click **Run evaluation**. The Flask backend retrains and reevaluates the models using those settings, then refreshes the metrics table and F1 chart. The included dataset is synthetic, so these metrics are for educational comparison only and do not establish real-world agricultural accuracy.

## Dynamic model evaluation and custom CSV

Open **Reports & farm data → Upload your own training CSV** to validate and activate a labelled dataset. The file must have at least 50 rows and these exact columns:

`soil_moisture, temperature, humidity, rainfall_mm, light_intensity, water_level, wind_speed, crop_type, irrigation_needed`

All seven numeric columns must contain finite numbers; `crop_type` must be filled; `irrigation_needed` must contain both binary labels `0` and `1` with at least five rows per class. Extra columns are ignored. The validated CSV is stored locally as `data/active_training_data.csv`; the original bundled demo CSV is not overwritten. Model Performance uses the active dataset. Choose a test split and model settings, then click **Run evaluation** to recompute metrics and the Random Forest confusion matrix. Use **Restore original demo dataset** to revert. Uploaded data persists across restarts until reset.

Use representative, lawfully obtained data and avoid uploading personal or confidential information. Synthetic/demo metrics are not evidence of real-world agronomic accuracy. This is a classroom prototype, not a field-validated irrigation prescription.
