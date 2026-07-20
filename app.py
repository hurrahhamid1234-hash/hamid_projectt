"""
============================================================
Phishing Website Detection — Flask App

CRITICAL FIXES applied (verified against dataset):

1. -1 convention matches training data exactly:
   - No directory/file in URL  → all directory & file features = -1
   - No query string in URL    → all params features = -1
   - Network lookup failed     → use legit-site median (NOT -1)

2. Network fallback values = median of legitimate-only rows
   (excluding -1 entries) from the training dataset.
   This prevents sites like google.com being misclassified.

3. Random Forest uses RAW features (no scaling).
   Logistic Regression uses StandardScaler output.
============================================================
"""

import re
import ssl
import socket
import threading
import datetime
import ipaddress

import joblib
import requests
import dns.resolver
import whois as pythonwhois
import pandas as pd
from urllib.parse import urlparse, parse_qs
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load model artefacts ──────────────────────────────────
MODEL      = joblib.load("models/best_phishing_model.pkl")
SCALER     = joblib.load("models/scaler.pkl")
FEAT_NAMES = joblib.load("models/feature_names.pkl")
MODEL_NAME = joblib.load("models/best_model_name.pkl")
print(f"[OK] Model    : {MODEL_NAME}")
print(f"[OK] Features : {len(FEAT_NAMES)}")

# ── Network fallback values ───────────────────────────────
# These are medians of LEGITIMATE-only rows (phishing=0)
# in the training dataset, excluding any -1 entries.
# Used when a live network lookup fails or times out.
NET_FALLBACK = {
    "time_response":          0.477,
    "domain_spf":             0.0,
    "asn_ip":                 22878.0,
    "time_domain_activation": 3299.0,
    "time_domain_expiration":  260.0,
    "qty_ip_resolved":         1.0,
    "qty_nameservers":         2.0,
    "qty_mx_servers":          1.0,
    "ttl_hostname":            1796.0,
    "tls_ssl_certificate":     1.0,
    "qty_redirects":           0.0,
    "url_google_index":        0.0,
    "domain_google_index":     0.0,
    "url_shortened":           0.0,
}

# ── Known URL shorteners ──────────────────────────────────
URL_SHORTENERS = {
    "bit.ly","goo.gl","tinyurl.com","ow.ly","t.co","buff.ly",
    "is.gd","cli.gs","digg.com","short.to","su.pr","ht.ly",
    "lnkd.in","rb.gy","cutt.ly","shorturl.at","tiny.cc",
    "qr.ae","bc.vc","snipurl.com","ff.im","twit.ac","x.co",
}

# ────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────

def _is_ip(domain):
    try:
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        return 0

def _vowels(s):
    return sum(1 for c in s.lower() if c in "aeiou")

def _char_counts(s):
    """All 17 special-character counts for one URL segment."""
    return {
        "dot":          s.count("."),
        "hyphen":       s.count("-"),
        "underline":    s.count("_"),
        "slash":        s.count("/"),
        "questionmark": s.count("?"),
        "equal":        s.count("="),
        "at":           s.count("@"),
        "and":          s.count("&"),
        "exclamation":  s.count("!"),
        "space":        s.count(" "),
        "tilde":        s.count("~"),
        "comma":        s.count(","),
        "plus":         s.count("+"),
        "asterisk":     s.count("*"),
        "hashtag":      s.count("#"),
        "dollar":       s.count("$"),
        "percent":      s.count("%"),
    }

def _put(f, key, val):
    """Set feature only if it exists in the saved feature list."""
    if key in f:
        f[key] = val

def _apply_segment(f, text, seg):
    """Write all 17 qty_*_<seg> features for a URL segment."""
    for char, val in _char_counts(text).items():
        _put(f, f"qty_{char}_{seg}", val)

def _set_segment_minus1(f, seg):
    """
    Set all features for a segment to -1.
    This matches the training data convention:
      -1 = segment does not exist in this URL
    """
    for char in ["dot","hyphen","underline","slash","questionmark","equal",
                 "at","and","exclamation","space","tilde","comma","plus",
                 "asterisk","hashtag","dollar","percent"]:
        _put(f, f"qty_{char}_{seg}", -1)

    # Length key differs per segment
    length_key = {
        "directory": "directory_length",
        "file":      "file_length",
        "params":    "params_length",
    }.get(seg)
    if length_key:
        _put(f, length_key, -1)

    # Extra params-only keys
    if seg == "params":
        _put(f, "tld_present_params", -1)
        _put(f, "qty_params",         -1)

# ────────────────────────────────────────────────────────
# NETWORK LOOKUP FUNCTIONS
# Each runs in its own daemon thread.
# On failure → writes None → caller uses NET_FALLBACK.
# ────────────────────────────────────────────────────────

def _net_response(url, R):
    try:
        t0 = datetime.datetime.now()
        requests.get(url, timeout=5, allow_redirects=True)
        R["time_response"] = round(
            (datetime.datetime.now() - t0).total_seconds(), 3)
    except Exception:
        R["time_response"] = None

def _net_spf(domain, R):
    try:
        ans = dns.resolver.resolve(domain, "TXT", lifetime=5)
        R["domain_spf"] = 1 if any(
            "v=spf1" in r.to_text().lower() for r in ans) else 0
    except Exception:
        R["domain_spf"] = None

def _net_asn(domain, R):
    try:
        ip = socket.gethostbyname(domain)
        s  = socket.create_connection(("whois.cymru.com", 43), timeout=5)
        s.sendall(f" -f {ip}\n".encode())
        data = b""
        while True:
            chunk = s.recv(1024)
            if not chunk:
                break
            data += chunk
        s.close()
        for line in data.decode(errors="ignore").strip().split("\n"):
            if "|" in line and not line.startswith("AS"):
                try:
                    R["asn_ip"] = int(line.split("|")[0].strip())
                    return
                except ValueError:
                    pass
        R["asn_ip"] = None
    except Exception:
        R["asn_ip"] = None

def _net_whois(domain, R):
    try:
        w   = pythonwhois.whois(domain)
        now = datetime.datetime.now()
        cd  = w.creation_date
        cd  = cd[0] if isinstance(cd, list) else cd
        R["time_domain_activation"] = (
            (now - cd).days if isinstance(cd, datetime.datetime) else None)
        ed  = w.expiration_date
        ed  = ed[0] if isinstance(ed, list) else ed
        R["time_domain_expiration"] = (
            (ed - now).days if isinstance(ed, datetime.datetime) else None)
    except Exception:
        R["time_domain_activation"] = None
        R["time_domain_expiration"] = None

def _net_dns(domain, R):
    for rtype, key in [("A","qty_ip_resolved"),
                       ("NS","qty_nameservers"),
                       ("MX","qty_mx_servers")]:
        try:
            R[key] = len(list(
                dns.resolver.resolve(domain, rtype, lifetime=5)))
        except Exception:
            R[key] = None

def _net_ttl(domain, R):
    try:
        ans = dns.resolver.resolve(domain, "A", lifetime=5)
        R["ttl_hostname"] = ans.rrset.ttl if ans.rrset else None
    except Exception:
        R["ttl_hostname"] = None

def _net_tls(domain, R):
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(
            socket.create_connection((domain, 443), timeout=5),
            server_hostname=domain
        ) as s:
            R["tls_ssl_certificate"] = 1 if s.getpeercert() else 0
    except Exception:
        R["tls_ssl_certificate"] = None

def _net_redirects(url, R):
    try:
        resp = requests.get(url, timeout=5, allow_redirects=True)
        R["qty_redirects"] = len(resp.history)
    except Exception:
        R["qty_redirects"] = None

def _net_google(url, domain, R):
    # Real check needs Google Search Console API.
    # 0 is the most common value in training data for both features.
    R["url_google_index"]    = 0
    R["domain_google_index"] = 0

def _net_shortened(domain, R):
    R["url_shortened"] = 1 if domain.lower() in URL_SHORTENERS else 0

# ────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTOR
# ────────────────────────────────────────────────────────

def extract_features(raw_url):
    # Start every feature at 0
    f = {col: 0 for col in FEAT_NAMES}

    # Ensure scheme so urlparse splits correctly
    url = (raw_url
           if raw_url.startswith(("http://", "https://"))
           else "http://" + raw_url)

    p         = urlparse(url)
    full_url  = url
    domain    = (p.netloc or "").split(":")[0]
    path      = p.path    or ""
    query_str = p.query   or ""

    # ── SEGMENT 1 : full URL ────────────────────────────
    _apply_segment(f, full_url, "url")
    _put(f, "qty_tld_url",
         sum(full_url.lower().count(t)
             for t in [".com",".net",".org",".info",".biz",
                       ".io",".gov",".edu",".co",".uk"]))
    _put(f, "length_url",   len(full_url))
    _put(f, "email_in_url",
         1 if re.search(
             r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
             full_url) else 0)

    # ── SEGMENT 2 : domain ──────────────────────────────
    _apply_segment(f, domain, "domain")
    _put(f, "qty_vowels_domain",    _vowels(domain))
    _put(f, "domain_length",        len(domain))
    _put(f, "domain_in_ip",         _is_ip(domain))
    _put(f, "server_client_domain",
         1 if re.search(r"\b(server|client)\b", domain, re.I) else 0)

    # ── SEGMENT 3 : directory  ───────────────────────────
    # Training convention: -1 when URL has no directory/file
    # path = /directory/file.ext
    # directory = everything before the last slash
    # file      = filename after  the last slash
    last_slash = path.rfind("/")

    if last_slash > 0:
        # URL has a real directory path
        directory = path[:last_slash]        # e.g. /admin/panel
        file_part = path[last_slash + 1:]    # e.g. login.php

        _apply_segment(f, directory, "directory")
        _put(f, "directory_length", len(directory))

        # ── SEGMENT 4 : file ────────────────────────────
        if file_part:
            _apply_segment(f, file_part, "file")
            _put(f, "file_length", len(file_part))
        else:
            # Directory exists but no filename after the slash
            _set_segment_minus1(f, "file")
    else:
        # URL has no path at all (e.g. www.google.com)
        # Both directory AND file are absent → -1
        _set_segment_minus1(f, "directory")
        _set_segment_minus1(f, "file")

    # ── SEGMENT 5 : params (query string) ───────────────
    # Training convention: -1 when URL has no query string
    if query_str:
        _apply_segment(f, query_str, "params")
        _put(f, "params_length", len(query_str))
        _put(f, "tld_present_params",
             1 if re.search(
                 r"\.(com|net|org|info|biz|io|co|gov|edu)",
                 query_str, re.I) else 0)
        _put(f, "qty_params", len(parse_qs(query_str)))
    else:
        # No query string → -1 for all params features
        _set_segment_minus1(f, "params")

    # ── SEGMENT 6 : network lookups (all parallel) ──────
    R = {}
    threads = [
        threading.Thread(target=_net_response,  args=(url, R)),
        threading.Thread(target=_net_spf,       args=(domain, R)),
        threading.Thread(target=_net_asn,       args=(domain, R)),
        threading.Thread(target=_net_whois,     args=(domain, R)),
        threading.Thread(target=_net_dns,       args=(domain, R)),
        threading.Thread(target=_net_ttl,       args=(domain, R)),
        threading.Thread(target=_net_tls,       args=(domain, R)),
        threading.Thread(target=_net_redirects, args=(url, R)),
        threading.Thread(target=_net_google,    args=(url, domain, R)),
        threading.Thread(target=_net_shortened, args=(domain, R)),
    ]
    for t in threads:
        t.daemon = True
        t.start()
    for t in threads:
        t.join(timeout=8)

    # Apply network results.
    # If a lookup returned None (failed) → use NET_FALLBACK
    # (legit-site medians). This is the key fix for google.com
    # being wrongly classified as phishing.
    for key, fallback_val in NET_FALLBACK.items():
        val = R.get(key)
        if val is None:
            val = fallback_val
            print(f"[FALLBACK] {key} = {fallback_val}")
        _put(f, key, val)

    return f

# ────────────────────────────────────────────────────────
# FLASK ROUTES
# ────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html", model_name=MODEL_NAME)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_url = (request.form.get("url") or "").strip()
        if not raw_url:
            return jsonify({"status": "error",
                            "message": "Please enter a URL."})

        print(f"\n{'='*55}")
        print(f"[SCAN] {raw_url}")

        feats = extract_features(raw_url)
        df    = pd.DataFrame([feats]).reindex(
            columns=FEAT_NAMES, fill_value=0)

        nz = {k: v for k, v in df.iloc[0].items() if v != 0}
        print(f"[FEATS] Non-zero: {len(nz)} / {len(FEAT_NAMES)}")

        # Random Forest → raw features (no scaling)
        # Logistic Regression → StandardScaler
        if MODEL_NAME == "Random Forest":
            pred = MODEL.predict(df)[0]
            prob = MODEL.predict_proba(df)[0]
        else:
            df_s = SCALER.transform(df)
            pred = MODEL.predict(df_s)[0]
            prob = MODEL.predict_proba(df_s)[0]

        label = "Phishing" if int(pred) == 1 else "Legitimate"
        print(f"[RESULT] {label}  "
              f"phish={prob[1]:.3f}  legit={prob[0]:.3f}")

        return jsonify({
            "status":                 "success",
            "url":                    raw_url,
            "prediction":             label,
            "phishing_probability":   round(float(prob[1]) * 100, 2),
            "legitimate_probability": round(float(prob[0]) * 100, 2),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False,
            host="0.0.0.0", port=5000)