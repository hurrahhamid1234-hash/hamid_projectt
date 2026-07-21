
"""
============================================================
Phishing Website Detection — Training Script
Dataset : 58,645 rows | 111 features | target: phishing
-1 convention:
  directory/file features = -1 when URL has no path
  params features         = -1 when URL has no query string
  network features        = -1 when lookup failed
============================================================
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# ── 111 feature columns in exact dataset order ────────────
FEATURE_COLS = [
    'qty_dot_url','qty_hyphen_url','qty_underline_url','qty_slash_url',
    'qty_questionmark_url','qty_equal_url','qty_at_url','qty_and_url',
    'qty_exclamation_url','qty_space_url','qty_tilde_url','qty_comma_url',
    'qty_plus_url','qty_asterisk_url','qty_hashtag_url','qty_dollar_url',
    'qty_percent_url','qty_tld_url','length_url',
    'qty_dot_domain','qty_hyphen_domain','qty_underline_domain',
    'qty_slash_domain','qty_questionmark_domain','qty_equal_domain',
    'qty_at_domain','qty_and_domain','qty_exclamation_domain',
    'qty_space_domain','qty_tilde_domain','qty_comma_domain',
    'qty_plus_domain','qty_asterisk_domain','qty_hashtag_domain',
    'qty_dollar_domain','qty_percent_domain','qty_vowels_domain',
    'domain_length','domain_in_ip','server_client_domain',
    'qty_dot_directory','qty_hyphen_directory','qty_underline_directory',
    'qty_slash_directory','qty_questionmark_directory','qty_equal_directory',
    'qty_at_directory','qty_and_directory','qty_exclamation_directory',
    'qty_space_directory','qty_tilde_directory','qty_comma_directory',
    'qty_plus_directory','qty_asterisk_directory','qty_hashtag_directory',
    'qty_dollar_directory','qty_percent_directory','directory_length',
    'qty_dot_file','qty_hyphen_file','qty_underline_file','qty_slash_file',
    'qty_questionmark_file','qty_equal_file','qty_at_file','qty_and_file',
    'qty_exclamation_file','qty_space_file','qty_tilde_file','qty_comma_file',
    'qty_plus_file','qty_asterisk_file','qty_hashtag_file','qty_dollar_file',
    'qty_percent_file','file_length',
    'qty_dot_params','qty_hyphen_params','qty_underline_params',
    'qty_slash_params','qty_questionmark_params','qty_equal_params',
    'qty_at_params','qty_and_params','qty_exclamation_params',
    'qty_space_params','qty_tilde_params','qty_comma_params',
    'qty_plus_params','qty_asterisk_params','qty_hashtag_params',
    'qty_dollar_params','qty_percent_params','params_length',
    'tld_present_params','qty_params','email_in_url',
    'time_response','domain_spf','asn_ip',
    'time_domain_activation','time_domain_expiration',
    'qty_ip_resolved','qty_nameservers','qty_mx_servers',
    'ttl_hostname','tls_ssl_certificate','qty_redirects',
    'url_google_index','domain_google_index','url_shortened',
]

TARGET = 'phishing'

# ── Load dataset ──────────────────────────────────────────
print("Loading dataset...")
data = pd.read_csv("dataset_small.csv")
print(f"Shape  : {data.shape}")
print(f"Target : {data[TARGET].value_counts().to_dict()}")

FEATURE_COLS = [c for c in FEATURE_COLS if c in data.columns]
print(f"Features used: {len(FEATURE_COLS)}")

X = data[FEATURE_COLS].copy()
y = data[TARGET].copy()

# ── Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ── Scale (for Logistic Regression) ──────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ════════════════════════════════════════════════════════
# MODEL 1 — Logistic Regression
# ════════════════════════════════════════════════════════
print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)
grid_lr = GridSearchCV(
    LogisticRegression(max_iter=2000),
    {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    },
    cv=5,
    n_jobs=1,
    verbose=1
)
grid_lr.fit(X_train_sc, y_train)
lr_model = grid_lr.best_estimator_
lr_cv    = cross_val_score(lr_model, X_train_sc, y_train, cv=5, n_jobs=-1).mean()
y_lr     = lr_model.predict(X_test_sc)
lr_acc   = accuracy_score(y_test, y_lr)
print(f"Best params   : {grid_lr.best_params_}")
print(f"CV accuracy   : {lr_cv:.4f}")
print(f"Test accuracy : {lr_acc:.4f}")

# ════════════════════════════════════════════════════════
# MODEL 2 — Random Forest
# ════════════════════════════════════════════════════════
print("\n" + "="*50)
print("RANDOM FOREST")
print("="*50)
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators":[100,200], "max_depth":[None,20]},
    cv=5, n_jobs=1, verbose=1)
grid_rf.fit(X_train, y_train)
rf_model = grid_rf.best_estimator_
rf_cv    = cross_val_score(rf_model, X_train, y_train, cv=5, n_jobs=1).mean()
y_rf     = rf_model.predict(X_test)
rf_acc   = accuracy_score(y_test, y_rf)
print(f"Best params   : {grid_rf.best_params_}")
print(f"CV accuracy   : {rf_cv:.4f}")
print(f"Test accuracy : {rf_acc:.4f}")

# ════════════════════════════════════════════════════════
# Pick best model
# ════════════════════════════════════════════════════════
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"Logistic Regression : {lr_acc:.4f}")
print(f"Random Forest       : {rf_acc:.4f}")

plt.figure(figsize=(6,4))
bars = plt.bar(["Logistic Regression","Random Forest"],
               [lr_acc, rf_acc], color=["#3b82f6","#10b981"], width=0.4)
for b in bars:
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
             f"{b.get_height():.4f}", ha="center", fontsize=11)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy"); plt.ylim(0.5,1.05)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150); plt.close()

if rf_acc >= lr_acc:
    best_model      = rf_model
    best_model_name = "Random Forest"
    y_pred          = y_rf
    best_X_test     = X_test        # RF uses raw (unscaled) features
else:
    best_model      = lr_model
    best_model_name = "Logistic Regression"
    y_pred          = y_lr
    best_X_test     = X_test_sc     # LR uses scaled features

print(f"\nBest model: {best_model_name}")

# ── Confusion matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax)
ax.set_xticks([0,1]); ax.set_xticklabels(["Legitimate","Phishing"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Legitimate","Phishing"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_model_name}")
thresh = cm.max()/2
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=14,
                color="white" if cm[i,j]>thresh else "black")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150); plt.close()
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate","Phishing"]))

# ── ROC Curve ─────────────────────────────────────────────
y_prob      = best_model.predict_proba(best_X_test)[:,1]
fpr,tpr,_   = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")
plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,color="#3b82f6",lw=2,label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {best_model_name}"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("plots/roc_curve.png", dpi=150); plt.close()

# ── Feature importance ────────────────────────────────────
plt.figure(figsize=(14,5))
if best_model_name == "Random Forest":
    imp = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
else:
    imp = pd.Series(lr_model.coef_[0], index=FEATURE_COLS).abs()
imp.sort_values(ascending=False).head(20).plot(kind="bar", color="#3b82f6")
plt.title(f"Top 20 Feature Importances ({best_model_name})")
plt.ylabel("Importance"); plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150); plt.close()

# ── Save everything ───────────────────────────────────────
joblib.dump(best_model,      "models/best_phishing_model.pkl")
joblib.dump(scaler,          "models/scaler.pkl")
joblib.dump(FEATURE_COLS,    "models/feature_names.pkl")
joblib.dump(best_model_name, "models/best_model_name.pkl")
joblib.dump({"lr_accuracy":lr_acc,"rf_accuracy":rf_acc,"roc_auc":roc_auc},
            "models/metrics.pkl")

print("\n" + "="*50)
print("ALL FILES SAVED")
print("="*50)
print(f"  models/best_phishing_model.pkl  ({best_model_name})")
print(f"  models/scaler.pkl")
print(f"  models/feature_names.pkl        ({len(FEATURE_COLS)} features)")
print(f"  models/best_model_name.pkl")
print(f"  models/metrics.pkl")
print(f"  plots/ (4 plots)")
