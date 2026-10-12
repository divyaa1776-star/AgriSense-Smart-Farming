from pathlib import Path
from datetime import datetime
import json, csv, heapq, math, sqlite3, os, secrets
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
TRAIN=DATA/"synthetic_training_data.csv"
SENSOR=DATA/"sensor_data.csv"
RUNS=DATA/"experiment_runs.csv"
ACTIVE_TRAIN=DATA/"active_training_data.csv"
USERS_DB=DATA/"agrisense_users.sqlite3"
app=Flask(__name__)
# Local development secret key; set AGRISENSE_SECRET_KEY in production.
_secret_file=ROOT/".flask_secret"
if _secret_file.exists():
    app.secret_key=_secret_file.read_text(encoding="utf-8").strip()
else:
    app.secret_key=os.environ.get("AGRISENSE_SECRET_KEY") or secrets.token_hex(32)
    if not os.environ.get("AGRISENSE_SECRET_KEY"):
        _secret_file.write_text(app.secret_key, encoding="utf-8")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

def db_connect():
    conn=sqlite3.connect(USERS_DB)
    conn.row_factory=sqlite3.Row
    return conn

def init_auth_db():
    DATA.mkdir(parents=True, exist_ok=True)
    with db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, language TEXT NOT NULL DEFAULT 'en', created_at TEXT NOT NULL)")
init_auth_db()

@app.before_request
def require_login():
    public_endpoints={"login", "register", "logout", "static"}
    if request.endpoint not in public_endpoints and not session.get("user_id"):
        if request.path.startswith("/api/"):
            return jsonify({"error":"Please sign in to use AgriSense."}), 401
        return redirect(url_for("login", next=request.path))

df=pd.read_csv(ACTIVE_TRAIN if ACTIVE_TRAIN.exists() else TRAIN)
sensor=pd.read_csv(SENSOR)
NUM=["soil_moisture","temperature","humidity","rainfall_mm","light_intensity","water_level","wind_speed"]
FEATURES=NUM+["crop_type"]
MODELS={
 "KNN":KNeighborsClassifier(n_neighbors=5),
 "Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42),
 "Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
 "Naive Bayes":GaussianNB(),
 "Random Forest":RandomForestClassifier(n_estimators=150,random_state=42)
}
def preprocessor(scale=True):
    ns=[("imputer",SimpleImputer(strategy="median"))]
    if scale: ns.append(("scaler",StandardScaler()))
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num",Pipeline(ns),NUM),("cat",cat,["crop_type"])])
def train_model(name):
    X=df[FEATURES]; y=df["irrigation_needed"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    pipe=Pipeline([("prep",preprocessor(name in ["KNN","Logistic Regression"])),("model",MODELS[name])])
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    return pipe,yte,pred
def safe(v):
    return float(v) if np.isfinite(v) else None

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))
    if request.method == "POST":
        email=request.form.get("email", "").strip().lower()
        password=request.form.get("password", "")
        with db_connect() as conn:
            user=conn.execute("SELECT id,name,email,password_hash,language FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"]=user["id"]
            session["user_name"]=user["name"]
            session["user_email"]=user["email"]
            session["language"]=user["language"]
            next_url=request.args.get("next", "")
            if not next_url.startswith("/") or next_url.startswith("//"):
                next_url=url_for("home")
            return redirect(next_url)
        flash("Email or password is incorrect.", "error")
    return render_template("auth.html", mode="login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))
    if request.method == "POST":
        name=request.form.get("name", "").strip()
        email=request.form.get("email", "").strip().lower()
        password=request.form.get("password", "")
        confirm=request.form.get("confirm_password", "")
        language=request.form.get("language", "en")
        if language not in ("en", "hi", "mr"):
            language="en"
        if len(name)<2 or len(name)>80:
            flash("Please enter a name between 2 and 80 characters.", "error")
        elif "@" not in email or len(email)>254:
            flash("Please enter a valid email address.", "error")
        elif len(password)<8:
            flash("Password must be at least 8 characters long.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        else:
            try:
                with db_connect() as conn:
                    cur=conn.execute("INSERT INTO users (name,email,password_hash,language,created_at) VALUES (?,?,?,?,?)",
                        (name,email,generate_password_hash(password),language,datetime.now().isoformat(timespec="seconds")))
                    uid=cur.lastrowid
                session.clear()
                session["user_id"]=uid
                session["user_name"]=name
                session["user_email"]=email
                session["language"]=language
                flash("Your AgriSense account is ready.", "success")
                return redirect(url_for("home"))
            except sqlite3.IntegrityError:
                flash("An account with this email already exists. Please sign in.", "error")
    return render_template("auth.html", mode="register")

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))

@app.route("/")
def home():
    return render_template("index.html",records=len(sensor),train_rows=len(df),avg_moisture=round(float(df.soil_moisture.mean()),1),
        avg_temp=round(float(df.temperature.mean()),1),low=int((df.soil_moisture<30).sum()),sensor_cols=list(sensor.columns),sensor_preview=sensor.head(8).fillna("").to_dict("records"), user_name=session.get("user_name", "Farmer"), user_email=session.get("user_email", ""), user_language=session.get("language", "en"))

@app.route("/api/predict",methods=["POST"])
def predict():
    d=request.get_json(force=True)
    name=d.get("model","Random Forest")
    if name not in MODELS: name="Random Forest"
    pipe,_,_=train_model(name)
    row=pd.DataFrame([{
      "soil_moisture":float(d["soil_moisture"]),"temperature":float(d["temperature"]),
      "humidity":float(d["humidity"]),"rainfall_mm":float(d["rainfall_mm"]),
      "light_intensity":float(d["light_intensity"]),"water_level":float(d["water_level"]),
      "wind_speed":float(d["wind_speed"]),"crop_type":d["crop_type"]
    }])
    ans=int(pipe.predict(row)[0]); prob=float(pipe.predict_proba(row)[0][1])
    # Transparent rules are supplementary and not claimed as agronomic truth.
    if float(d["rainfall_mm"])>=8: advice="Rainfall input is high. Recheck soil moisture after rainfall before irrigating."
    elif float(d["water_level"])<=15: advice="Available water is low. Prioritize conservation and inspect the water system."
    elif float(d["soil_moisture"])<25: advice="Soil moisture is low. Consider irrigation after checking crop stage and field conditions."
    elif ans: advice="The selected model flags possible irrigation need. Verify field conditions before acting."
    else: advice="The selected model does not flag immediate irrigation need. Continue monitoring."
    return jsonify({"model":name,"prediction":ans,"probability":round(prob,4),"advice":advice,
                    "note":"Prediction is trained on synthetic proxy labels; it is a classroom demonstration, not field-validated advice."})

@app.route("/api/models", methods=["GET", "POST"])
def model_metrics():
    """Evaluate models with user-selected settings; metrics are recalculated from data."""
    settings = request.get_json(silent=True) or {}
    try:
        test_size = float(settings.get("test_size", 0.25))
        knn_k = int(settings.get("knn_k", 5))
        tree_depth = int(settings.get("tree_depth", 5))
        forest_trees = int(settings.get("forest_trees", 150))
    except (TypeError, ValueError):
        return jsonify({"error": "Settings must be valid numbers."}), 400
    if not 0.1 <= test_size <= 0.4:
        return jsonify({"error": "Test split must be between 10% and 40%."}), 400
    if not 1 <= knn_k <= 25 or not 1 <= tree_depth <= 20 or not 10 <= forest_trees <= 500:
        return jsonify({"error": "KNN k must be 1–25, tree depth 1–20 and forest trees 10–500."}), 400

    X=df[FEATURES]; y=df.irrigation_needed
    try:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)
    except ValueError as exc:
        return jsonify({"error": f"Could not create train/test split: {exc}"}), 400

    effective_knn_k = min(knn_k, len(ytr))
    eval_models={
        "KNN": KNeighborsClassifier(n_neighbors=effective_knn_k),
        "Decision Tree": DecisionTreeClassifier(max_depth=tree_depth,random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=forest_trees,random_state=42),
    }
    rows=[]; forest_pred=None
    for name, estimator in eval_models.items():
        pipe=Pipeline([("prep",preprocessor(name in ["KNN", "Logistic Regression"])),("model",estimator)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        rows.append({"model":name,"accuracy":safe(accuracy_score(yte,pred)),
          "precision":safe(precision_score(yte,pred,zero_division=0)),
          "recall":safe(recall_score(yte,pred,zero_division=0)),
          "f1":safe(f1_score(yte,pred,zero_division=0))})
        if name == "Random Forest": forest_pred=pred

    base=Pipeline([("prep",preprocessor()),("model",DummyClassifier(strategy="most_frequent"))]).fit(Xtr,ytr)
    bpred=base.predict(Xte)
    cv_pipeline=Pipeline([("prep",preprocessor()),("model",RandomForestClassifier(n_estimators=forest_trees,random_state=42))])
    cv= cross_val_score(cv_pipeline,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring="f1")
    return jsonify({"models":rows,"baseline_f1":safe(f1_score(yte,bpred,zero_division=0)),
        "cv_mean":safe(cv.mean()),"cv_std":safe(cv.std()),
        "confusion_matrix":confusion_matrix(yte,forest_pred).tolist(),
        "settings":{"test_size":test_size,"knn_k":effective_knn_k,"requested_knn_k":knn_k,"tree_depth":tree_depth,"forest_trees":forest_trees},
        "dataset_rows":int(len(df)), "dataset_source":"uploaded dataset" if ACTIVE_TRAIN.exists() else "included synthetic demo dataset",
        "note":"Metrics were recalculated from the active dataset. If it is synthetic or rule-labelled, results are educational and are not field-validated agricultural performance."})

@app.route("/api/clusters")
def clusters():
    algo=request.args.get("algo","K-means")
    cols=NUM
    X=np.nan_to_num(df[cols].to_numpy(),nan=0.0)
    X=StandardScaler().fit_transform(X)
    if algo=="DBSCAN": labels=DBSCAN(eps=1.1,min_samples=8).fit_predict(X)
    elif algo=="Hierarchical": labels=AgglomerativeClustering(n_clusters=3).fit_predict(X)
    else: labels=KMeans(n_clusters=3,n_init=10,random_state=42).fit_predict(X)
    coords=PCA(n_components=2).fit_transform(X)
    points=[{"x":round(float(a),3),"y":round(float(b),3),"cluster":str(int(c)),"moisture":float(m)} for a,b,c,m in zip(coords[:,0],coords[:,1],labels,df.soil_moisture)]
    return jsonify({"algorithm":algo,"points":points,"counts":pd.Series(labels).value_counts().sort_index().to_dict()})

@app.route("/api/schedule",methods=["POST"])
def schedule():
    d=request.get_json(force=True); days=int(d.get("days",7)); demand=float(d.get("demand",15)); budget=float(d.get("budget",18))
    days=max(3,min(14,days)); budget=max(0,min(35,budget)); demand=max(0,min(30,demand))
    def obj(s):
        s=np.array(s); return float(np.sum(np.maximum(demand-s,0)**2+.15*np.maximum(s-demand,0)**2)+.2*np.sum(s**2))
    rng=np.random.default_rng(42); start=[min(budget,demand//2)]*days
    # Hill climbing
    cur=start[:]; best=cur[:]; bs=obj(best)
    for _ in range(500):
        cand=cur[:]; j=int(rng.integers(days)); cand[j]=int(np.clip(cand[j]+rng.choice([-4,-2,-1,1,2,4]),0,budget))
        if obj(cand)<obj(cur): cur=cand
        if obj(cur)<bs: best=cur[:]; bs=obj(best)
    # Simulated annealing
    cur=start[:]; cs=obj(cur); bestsa=cur[:]; bss=cs
    for i in range(1000):
        t=max(.01,10*(.995**i)); cand=cur[:]; j=int(rng.integers(days)); cand[j]=int(np.clip(cand[j]+rng.choice([-5,-2,-1,1,2,5]),0,budget)); ns=obj(cand)
        if ns<cs or rng.random()<np.exp(min(0,(cs-ns)/t)): cur,cs=cand,ns
        if cs<bss: bestsa,bss=cur[:],cs
    return jsonify({"days":days,"hill":best,"annealing":bestsa,"hill_score":round(bs,2),"annealing_score":round(bss,2),
                    "note":"Illustrative simulation only; validate crop and soil conditions before real irrigation."})


@app.route("/api/lab/regression", methods=["POST"])
def lab_regression():
    # Predict synthetic soil moisture from environmental features; educational only.
    features = ["temperature", "humidity", "rainfall_mm", "light_intensity", "water_level", "wind_speed"]
    X = df[features].copy()
    y = df["soil_moisture"].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LinearRegression())])
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return jsonify({"mae": safe(mean_absolute_error(yte, pred)), "rmse": safe(float(np.sqrt(mean_squared_error(yte, pred)))),
                    "r2": safe(r2_score(yte, pred)), "actual": yte.iloc[:40].round(2).tolist(), "predicted": np.round(pred[:40],2).tolist(),
                    "features": features, "coefficients": np.round(model.named_steps['model'].coef_, 4).tolist(),
                    "note": "Regression targets come from synthetic training data and are not field-calibrated."})

@app.route("/api/lab/ann", methods=["POST"])
def lab_ann():
    X = df[FEATURES]; y = df["irrigation_needed"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    hidden = int((request.get_json(silent=True) or {}).get("hidden", 32))
    hidden = max(8, min(96, hidden))
    pipe = Pipeline([("prep", preprocessor(True)), ("ann", MLPClassifier(hidden_layer_sizes=(hidden, 16), activation="relu", solver="adam", max_iter=500, random_state=42, early_stopping=True))])
    pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)
    return jsonify({"accuracy": safe(accuracy_score(yte,pred)), "precision": safe(precision_score(yte,pred,zero_division=0)),
                    "recall": safe(recall_score(yte,pred,zero_division=0)), "f1": safe(f1_score(yte,pred,zero_division=0)),
                    "iterations": int(pipe.named_steps['ann'].n_iter_), "hidden": hidden,
                    "note": "ANN uses an MLPClassifier with ReLU hidden layers and Adam optimizer. Results use synthetic proxy labels."})

@app.route("/api/lab/apriori", methods=["POST"])
def lab_apriori():
    # Build interpretable transactions from environmental conditions and frequent-itemset counts.
    payload = request.get_json(silent=True) or {}
    min_support = float(payload.get("support", 0.15)); min_support = min(.8, max(.05, min_support))
    transactions=[]
    for _, r in df.iterrows():
        items=[]
        items.append("Low moisture" if r.soil_moisture < 30 else "Adequate moisture")
        items.append("Hot" if r.temperature >= 32 else "Moderate temperature")
        items.append("Rain expected" if r.rainfall_mm >= 8 else "Low rainfall")
        items.append("Low water reserve" if r.water_level < 25 else "Water reserve available")
        items.append("Irrigation flagged" if int(r.irrigation_needed)==1 else "Irrigation not flagged")
        transactions.append(set(items))
    counts=Counter()
    for t in transactions:
        for k in (1,2,3):
            for combo in combinations(sorted(t), k): counts[combo]+=1
    threshold=int(np.ceil(min_support*len(transactions)))
    rows=[]
    for itemset,count in counts.items():
        if count>=threshold:
            rows.append({"items": list(itemset), "count": count, "support": round(count/len(transactions),3)})
    rows.sort(key=lambda x:(-len(x['items']),-x['support'],x['items']))
    return jsonify({"transactions":len(transactions),"min_support":min_support,"itemsets":rows[:80],
                    "note":"Apriori-style exhaustive itemset counting on a small set of discretized synthetic attributes; support is descriptive, not causal."})

@app.route("/api/lab/search", methods=["POST"])
def lab_search():
    payload=request.get_json(silent=True) or {}
    algo=payload.get("algorithm","A*")
    # Small grid path-planning demo. 1 = blocked; 0 = traversable.
    grid=[[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,0,0,0,0,1,0],[1,1,0,1,0,0,0],[0,0,0,0,0,1,0],[0,1,1,1,0,0,0],[0,0,0,1,0,0,0]]
    start=(0,0); goal=(6,6); frontier=[(0,start)]; came={start:None}; cost={start:0}; expanded=0
    import heapq
    while frontier:
        _,cur=heapq.heappop(frontier); expanded+=1
        if cur==goal: break
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nxt=(cur[0]+dr,cur[1]+dc)
            if not(0<=nxt[0]<7 and 0<=nxt[1]<7) or grid[nxt[0]][nxt[1]]: continue
            nc=cost[cur]+1
            if nxt not in cost or nc<cost[nxt]:
                cost[nxt]=nc; came[nxt]=cur
                h=abs(goal[0]-nxt[0])+abs(goal[1]-nxt[1]) if algo=="A*" else 0
                heapq.heappush(frontier,(nc+h,nxt))
    path=[]; cur=goal
    if cur in came:
        while cur is not None: path.append(list(cur)); cur=came[cur]
        path.reverse()
    return jsonify({"algorithm":algo,"grid":grid,"path":path,"path_length":max(0,len(path)-1),"expanded_nodes":expanded,
                    "note":"A small grid search demonstration. The grid is illustrative and not a real farm map."})


@app.route("/api/lab/optimization", methods=["POST"])
def lab_optimization():
    payload=request.get_json(silent=True) or {}
    days=max(3,min(14,int(payload.get("days",7))))
    demand=max(1,min(30,float(payload.get("demand",12))))
    budget=max(1,min(35,float(payload.get("daily_budget",18))))
    seed=int(payload.get("seed",42)); rng=np.random.default_rng(seed)
    # Objective penalizes unmet demand, overwatering, and abrupt day-to-day changes.
    def score(plan):
        a=np.asarray(plan,dtype=float)
        unmet=np.maximum(demand-a,0)
        excess=np.maximum(a-demand,0)
        change=np.diff(a)
        return float(np.sum(2.5*unmet**2 + .45*excess**2 + .12*change**2))
    initial=np.full(days, min(budget,demand),dtype=int)
    # Hill climbing
    cur=initial.copy(); best=cur.copy(); best_score=score(best)
    for _ in range(1800):
        cand=cur.copy(); j=int(rng.integers(days)); cand[j]=int(np.clip(cand[j]+rng.choice([-5,-3,-1,1,3,5]),0,budget))
        if score(cand)<score(cur): cur=cand
        if score(cur)<best_score: best=cur.copy(); best_score=score(best)
    # Simulated annealing
    cur=initial.copy(); cur_score=score(cur); sa=cur.copy(); sa_score=cur_score
    for i in range(3000):
        temp=max(.02,8*(.997**i)); cand=cur.copy(); j=int(rng.integers(days)); cand[j]=int(np.clip(cand[j]+rng.choice([-5,-3,-1,1,3,5]),0,budget)); cs=score(cand)
        if cs<cur_score or rng.random()<math.exp(min(0,(cur_score-cs)/temp)): cur,cur_score=cand,cs
        if cur_score<sa_score: sa,sa_score=cur.copy(),cur_score
    return jsonify({"days":days,"demand":demand,"daily_budget":budget,"hill_climbing":best.tolist(),"simulated_annealing":sa.tolist(),"baseline":initial.tolist(),"scores":{"baseline":round(score(initial),2),"hill_climbing":round(best_score,2),"simulated_annealing":round(sa_score,2)},"note":"Toy objective for algorithm comparison, not an agronomic water prescription. Units are illustrative."})

@app.route("/api/lab/classical-search", methods=["POST"])
def classical_search():
    payload=request.get_json(silent=True) or {}; algo=payload.get("algorithm","BFS")
    graph={"Farm": [("North field",2),("Pump",3)], "North field":[("Reservoir",4),("Pump",1)], "Pump":[("Reservoir",2),("South field",5)], "Reservoir":[("South field",2)], "South field":[]}
    start="Farm"; goal="South field"; expanded=0; parent={start:None}; cost={start:0}
    if algo in ("BFS","DFS"):
        frontier=[start]; seen={start}
        while frontier:
            cur=frontier.pop(0) if algo=="BFS" else frontier.pop(); expanded+=1
            if cur==goal: break
            neigh=[n for n,_ in graph[cur]]
            for nxt in (neigh if algo=="BFS" else list(reversed(neigh))):
                if nxt not in seen: seen.add(nxt); parent[nxt]=cur; cost[nxt]=cost[cur]+1; frontier.append(nxt)
    else:
        q=[(0,start)]
        while q:
            c,cur=heapq.heappop(q); expanded+=1
            if c!=cost[cur]: continue
            if cur==goal: break
            for nxt,w in graph[cur]:
                nc=c+w
                if nxt not in cost or nc<cost[nxt]: cost[nxt]=nc; parent[nxt]=cur; heapq.heappush(q,(nc,nxt))
    path=[]; cur=goal
    if cur in parent:
        while cur is not None: path.append(cur); cur=parent[cur]
        path.reverse()
    return jsonify({"algorithm":algo,"graph":graph,"path":path,"cost":cost.get(goal),"expanded_nodes":expanded,"note":"Graph is a small illustrative farm-routing example. BFS/DFS optimize hop count or traversal order; UCS optimizes edge cost."})

@app.route("/api/lab/tsp", methods=["POST"])
def lab_tsp():
    # Coordinates represent fictional farm inspection points.
    pts=np.array([[0,0],[2,5],[5,4],[6,1],[3,2],[1,3]],dtype=float); names=["Pump","North plot","Reservoir","South plot","Store","Gate"]
    dist=lambda a,b: float(np.linalg.norm(pts[a]-pts[b]))
    route=list(range(len(pts))); rng=np.random.default_rng(42)
    def length(r): return sum(dist(r[i],r[(i+1)%len(r)]) for i in range(len(r)))
    # Nearest neighbour baseline
    nn=[0]; remaining=set(range(1,len(pts)))
    while remaining:
        nxt=min(remaining,key=lambda j:dist(nn[-1],j)); nn.append(nxt); remaining.remove(nxt)
    # Simulated annealing over tour permutations, preserve start at node 0
    cur=nn[:]; best=cur[:]; bs=length(best)
    for i in range(3500):
        temp=max(.01,3*(.997**i)); cand=cur[:]; a,b=sorted(rng.choice(np.arange(1,len(pts)),2,replace=False).tolist()); cand[a:b+1]=reversed(cand[a:b+1]); old=length(cur); new=length(cand)
        if new<old or rng.random()<math.exp(min(0,(old-new)/temp)): cur=cand
        if length(cur)<bs: best=cur[:]; bs=length(best)
    return jsonify({"points":[{"name":names[i],"x":float(pts[i,0]),"y":float(pts[i,1])} for i in range(len(pts))],"nearest_neighbor":[names[i] for i in nn]+[names[nn[0]]],"optimized_route":[names[i] for i in best]+[names[best[0]]],"nearest_distance":round(length(nn),2),"optimized_distance":round(bs,2),"note":"Synthetic coordinates and Euclidean distance; demonstrates TSP heuristics, not a surveyed field map."})

@app.route("/api/lab/csp", methods=["POST"])
def lab_csp():
    # CSP: assign irrigation windows to zones; adjacent zones cannot use same slot, pump capacity <= 2 zones per slot.
    zones=["North","South","East","West"]; slots=["06:00","08:00","17:00"]
    neighbors={"North":{"South","East"},"South":{"North","West"},"East":{"North","West"},"West":{"South","East"}}
    assigned={}; nodes=0
    def backtrack():
        nonlocal nodes
        nodes+=1
        if len(assigned)==len(zones): return True
        # MRV: most constrained unassigned zone
        un=[z for z in zones if z not in assigned]
        z=max(un,key=lambda q:sum(n in assigned for n in neighbors[q]))
        for slot in slots:
            if any(assigned.get(n)==slot for n in neighbors[z]): continue
            if sum(v==slot for v in assigned.values())>=2: continue
            assigned[z]=slot
            if backtrack(): return True
            del assigned[z]
        return False
    solved=backtrack()
    return jsonify({"solved":solved,"assignment":assigned,"constraints":["Adjacent zones cannot share a time slot","At most two zones per time slot"],"search_nodes":nodes,"note":"Constraint-satisfaction demonstration with fictional zones and time slots; actual pump flow and crop windows must be measured."})

@app.route("/api/lab/water-jug", methods=["POST"])
def lab_water_jug():
    payload=request.get_json(silent=True) or {}; a=max(1,min(12,int(payload.get("a",4)))); b=max(1,min(12,int(payload.get("b",3)))); target=max(0,min(max(a,b),int(payload.get("target",2))))
    start=(0,0); q=[start]; parent={start:None}; action={}; goal=None
    while q:
        x,y=q.pop(0)
        if x==target or y==target: goal=(x,y); break
        nxt=[((a,y),"Fill jug A"),((x,b),"Fill jug B"),((0,y),"Empty jug A"),((x,0),"Empty jug B"),((x-min(x,b-y),y+min(x,b-y)),"Pour A into B"),((x+min(y,a-x),y-min(y,a-x)),"Pour B into A")]
        for state,act in nxt:
            if state not in parent: parent[state]= (x,y); action[state]=act; q.append(state)
    steps=[]
    if goal:
        cur=goal
        while parent[cur] is not None: steps.append({"state":list(cur),"action":action[cur]}); cur=parent[cur]
        steps.reverse()
    return jsonify({"solved":goal is not None,"jug_capacities":[a,b],"target":target,"steps":steps,"states_explored":len(parent),"note":"Breadth-first search solves the classic Water Jug state-space problem."})

@app.route("/api/lab/game-search", methods=["POST"])
def game_search():
    # Minimax with alpha-beta pruning on a small illustrative game tree.
    tree={"A":["B","C"],"B":["D","E"],"C":["F","G"]}; values={"D":3,"E":5,"F":2,"G":9}; visited=0
    def minimax(node,maximizing,alpha,beta):
        nonlocal visited; visited+=1
        if node in values: return values[node]
        children=tree[node]
        if maximizing:
            value=-float('inf')
            for child in children:
                value=max(value,minimax(child,False,alpha,beta)); alpha=max(alpha,value)
                if beta<=alpha: break
            return value
        value=float('inf')
        for child in children:
            value=min(value,minimax(child,True,alpha,beta)); beta=min(beta,value)
            if beta<=alpha: break
        return value
    value=minimax("A",True,-float('inf'),float('inf'))
    return jsonify({"algorithm":"Minimax + alpha-beta pruning","root_value":value,"leaf_values":values,"visited_nodes":visited,"note":"Toy adversarial search tree for AIML algorithm viva; not used to make agricultural decisions."})



@app.route("/api/lab/vision", methods=["POST"])
def lab_vision():
    from PIL import Image
    from io import BytesIO
    file=request.files.get("image")
    if not file: return jsonify({"error":"Choose an image first."}),400
    try:
        im=Image.open(BytesIO(file.read())).convert("RGB")
        im.thumbnail((800,800)); arr=np.asarray(im,dtype=np.uint8)
        r=arr[:,:,0].astype(float); g=arr[:,:,1].astype(float); b=arr[:,:,2].astype(float)
        green=(g>r*1.08)&(g>b*1.05)&(g>45)
        brightness=float(arr.mean()/255*100)
        return jsonify({"width":int(arr.shape[1]),"height":int(arr.shape[0]),"brightness_percent":round(brightness,2),"green_pixel_ratio_percent":round(float(green.mean()*100),2),"mean_rgb":[round(float(r.mean()),1),round(float(g.mean()),1),round(float(b.mean()),1)],"note":"Basic image-processing heuristic only. Green-pixel ratio is not a crop-health diagnosis; lighting, soil, crop type and camera conditions affect it. This is not YOLO object detection."})
    except Exception:
        return jsonify({"error":"Could not read this image. Use PNG, JPG or WEBP."}),400

@app.route("/api/lab/semi-supervised", methods=["POST"])
def lab_semi_supervised():
    # Demonstration: train on a small labeled subset, pseudo-label remaining training rows, refit.
    from sklearn.semi_supervised import SelfTrainingClassifier
    X=df[FEATURES]; y=df["irrigation_needed"].to_numpy().copy()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    # SelfTrainingClassifier validates its input before calling the base estimator,
    # so categorical crop_type values must be encoded before fitting the wrapper.
    prep=preprocessor(True)
    Xtr_enc=prep.fit_transform(Xtr)
    Xte_enc=prep.transform(Xte)
    rng=np.random.default_rng(42); mask=rng.random(len(ytr))<.25
    # Keep at least one labelled example from every class.
    for cls in np.unique(ytr):
        inds=np.where(ytr==cls)[0]
        if not mask[inds].any(): mask[inds[0]]=True
    yl=ytr.copy(); yl[~mask]=-1
    base=RandomForestClassifier(n_estimators=100,random_state=42)
    model=SelfTrainingClassifier(base,threshold=.8,max_iter=5)
    model.fit(Xtr_enc,yl); pred=model.predict(Xte_enc)
    return jsonify({"initially_labeled":int(mask.sum()),"initially_unlabeled":int((~mask).sum()),"pseudo_labeled":int(np.sum(model.transduction_[~mask]!=-1)),"accuracy":safe(accuracy_score(yte,pred)),"precision":safe(precision_score(yte,pred,zero_division=0)),"recall":safe(recall_score(yte,pred,zero_division=0)),"f1":safe(f1_score(yte,pred,zero_division=0)),"note":"Self-training pseudo-labels unlabeled synthetic rows; results are pedagogical and not evidence of field accuracy."})

@app.route("/api/lab/rl", methods=["POST"])
def lab_rl():
    # Tiny Q-learning environment: states represent low/adequate moisture; actions wait/irrigate.
    rng=np.random.default_rng(42); q=np.zeros((3,2)); episodes=500; alpha=.15; gamma=.9; eps=.2
    # state 0=very dry, 1=moderate, 2=wet; action 0=wait, 1=irrigate
    for _ in range(episodes):
        state=int(rng.choice([0,1,2],p=[.4,.4,.2]))
        for _step in range(8):
            action=int(rng.integers(2) if rng.random()<eps else np.argmax(q[state]))
            if state==0 and action==1: reward=4; nxt=1
            elif state==0 and action==0: reward=-4; nxt=0 if rng.random()<.7 else 1
            elif state==1 and action==1: reward=-2; nxt=2
            elif state==1 and action==0: reward=2; nxt=0 if rng.random()<.15 else 1
            elif state==2 and action==1: reward=-5; nxt=2
            else: reward=1; nxt=1 if rng.random()<.25 else 2
            q[state,action]+=alpha*(reward+gamma*np.max(q[nxt])-q[state,action]); state=nxt
    labels=["Very dry","Moderate","Wet"]
    policy={labels[i]:("Irrigate" if int(np.argmax(q[i]))==1 else "Wait / monitor") for i in range(3)}
    return jsonify({"episodes":episodes,"q_values":np.round(q,3).tolist(),"policy":policy,"note":"Toy Q-learning reward environment for RL coursework. Rewards and transitions are invented; do not use as farm control."})

@app.route("/api/lab/ao-star", methods=["POST"])
def lab_ao_star():
    # Small AND-OR graph: action A needs both soil check and weather check; action B uses reservoir check.
    graph={"Start":{"OR":[("Plan A",2),("Plan B",4)]},"Plan A":{"AND":[("Soil check",1),("Weather check",2)]},"Plan B":{"AND":[("Reservoir check",2),("Pump check",2)]},"Soil check":{"cost":1},"Weather check":{"cost":2},"Reservoir check":{"cost":2},"Pump check":{"cost":2}}
    # Calculate costs for the fixed small graph; AND children sum, OR selects minimum.
    cost={"Soil check":1,"Weather check":2,"Reservoir check":2,"Pump check":2}
    a=cost["Soil check"]+cost["Weather check"]+2
    b=cost["Reservoir check"]+cost["Pump check"]+4
    return jsonify({"graph":graph,"estimated_costs":{"Plan A":a,"Plan B":b},"selected_plan":"Plan A" if a<=b else "Plan B","selected_cost":min(a,b),"note":"AO* is shown as a small hand-evaluated AND-OR graph. It is an educational trace, not a learned farm planner."})

@app.route("/api/lab/experiment-log", methods=["GET","POST"])
def experiment_log():
    if request.method=="POST":
        d=request.get_json(silent=True) or {}
        row={"timestamp":datetime.now().isoformat(timespec="seconds"),"experiment":str(d.get("experiment","Manual experiment"))[:100],"algorithm":str(d.get("algorithm","unspecified"))[:100],"metric":str(d.get("metric",""))[:50],"value":str(d.get("value",""))[:50],"notes":str(d.get("notes",""))[:300]}
        new=not RUNS.exists()
        with RUNS.open("a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(row));
            if new: w.writeheader()
            w.writerow(row)
        return jsonify({"saved":True,"run":row})
    if RUNS.exists():
        rows=pd.read_csv(RUNS).fillna("").tail(100).to_dict("records")
    else: rows=[]
    return jsonify({"runs":rows,"count":len(rows)})

@app.route("/api/data-profile")
def data_profile():
    numeric=df.select_dtypes(include=np.number)
    return jsonify({"rows":int(len(df)),"columns":list(df.columns),"missing":{k:int(v) for k,v in df.isna().sum().items()},"summary":{c:{"mean":safe(numeric[c].mean()),"std":safe(numeric[c].std()),"min":safe(numeric[c].min()),"max":safe(numeric[c].max())} for c in numeric.columns},"target_balance":{str(k):int(v) for k,v in df.irrigation_needed.value_counts().to_dict().items()},"note":"Profiling describes the included synthetic dataset; it does not validate agronomic quality."})


@app.route("/api/sensor")
def sensor_data():
    return jsonify({"columns":list(sensor.columns),"rows":sensor.head(50).fillna("").to_dict("records"),"count":len(sensor)})

@app.route("/api/dataset/status")
def dataset_status():
    counts = df["irrigation_needed"].value_counts().to_dict() if "irrigation_needed" in df else {}
    return jsonify({"rows": int(len(df)), "source": "uploaded dataset" if ACTIVE_TRAIN.exists() else "included synthetic demo dataset", "columns": list(df.columns), "target_counts": {str(k): int(v) for k,v in counts.items()}})

@app.route("/api/dataset/upload", methods=["POST"])
def upload_training_dataset():
    """Validate and activate a user CSV for retraining/evaluation; keep original demo data intact."""
    global df
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename or not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"error": "Choose a .csv file."}), 400
    try:
        candidate = pd.read_csv(uploaded)
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV: {exc}"}), 400
    required = NUM + ["crop_type", "irrigation_needed"]
    missing = [c for c in required if c not in candidate.columns]
    if missing:
        return jsonify({"error": "Missing required columns: " + ", ".join(missing)}), 400
    if len(candidate) < 50:
        return jsonify({"error": "Upload at least 50 rows so train/test split and 5-fold validation are meaningful."}), 400
    candidate = candidate[required].copy()
    for col in NUM:
        candidate[col] = pd.to_numeric(candidate[col], errors="coerce")
    candidate["crop_type"] = candidate["crop_type"].astype("string").str.strip()
    candidate["irrigation_needed"] = pd.to_numeric(candidate["irrigation_needed"], errors="coerce")
    if candidate[NUM].isna().any().any() or not np.isfinite(candidate[NUM].to_numpy(dtype=float)).all():
        return jsonify({"error": "Numeric feature columns must contain valid finite numbers without missing values."}), 400
    if candidate["crop_type"].isna().any() or (candidate["crop_type"] == "").any():
        return jsonify({"error": "crop_type must be filled for every row."}), 400
    if candidate["irrigation_needed"].isna().any() or not set(candidate["irrigation_needed"].unique()).issubset({0,1}):
        return jsonify({"error": "irrigation_needed must contain only 0 or 1 labels."}), 400
    candidate["irrigation_needed"] = candidate["irrigation_needed"].astype(int)
    counts = candidate["irrigation_needed"].value_counts()
    if len(counts) != 2 or int(counts.min()) < 5:
        return jsonify({"error": "The target must contain both classes 0 and 1, with at least 5 records per class."}), 400
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        temp_path = ACTIVE_TRAIN.with_suffix(".tmp.csv")
        candidate.to_csv(temp_path, index=False)
        temp_path.replace(ACTIVE_TRAIN)
        df = candidate
    except Exception as exc:
        return jsonify({"error": f"Could not save dataset: {exc}"}), 500
    return jsonify({"message": "Dataset validated and activated. Model evaluation now uses this dataset.", "rows": int(len(df)), "columns": list(df.columns), "target_counts": {str(k): int(v) for k,v in counts.to_dict().items()}, "source": "uploaded dataset"})

@app.route("/api/dataset/reset", methods=["POST"])
def reset_training_dataset():
    global df
    try:
        if ACTIVE_TRAIN.exists():
            ACTIVE_TRAIN.unlink()
        df = pd.read_csv(TRAIN)
    except Exception as exc:
        return jsonify({"error": f"Could not restore included demo dataset: {exc}"}), 500
    return jsonify({"message": "Restored the original included synthetic demo dataset.", "rows": int(len(df)), "source": "included synthetic demo dataset"})

@app.route("/download/synthetic")
def download_data():
    return send_file(TRAIN,as_attachment=True,download_name="synthetic_training_data.csv")

if __name__=="__main__":
    app.run(debug=True,port=5000)
