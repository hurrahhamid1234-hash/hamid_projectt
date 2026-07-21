
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

4. RISK-LAYER FIXES (this revision):
   - '@' / '\' in the domain is now flagged directly
     (classic userinfo/host-spoofing trick, e.g.
     "bankdetails\otp@evil.com" where the real host is
     "evil.com", not "bankdetails\otp").
   - New subdomain-stacking rule: 2+ sensitive keywords
     each occupying their own subdomain label
     (e.g. bank.verify.account.evil-domain.com) is now
     flagged — this pattern is rare on legitimate sites.
   - Keyword-density cap raised from 0.36 to 0.65. The old
     cap made it mathematically impossible for keyword
     density alone to ever cross a 0.5 classification
     threshold, no matter how keyword-stuffed the URL was.
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

# ── Brand-impersonation / typosquat detection ─────────────
# The trained ML model only ever saw lexical counts + network
# stats — it was never given the URL text itself, so it has
# no way to recognise "g00gle.com", "rnicrosoft.com", or
# "paypal-login-secure.net" as impersonation attempts. This
# layer catches that class of attack independently and blends
# a weighted risk score into the model's verdict.

BRANDS = [
    "google","paypal","amazon","microsoft","apple","facebook",
    "instagram","netflix","bankofamerica","wellsfargo","chase",
    "citibank","hsbc","linkedin","ebay","twitter","whatsapp",
    "dropbox","adobe","yahoo","outlook","icloud","gmail",
]

SENSITIVE_KEYWORDS = [
    "login","signin","sign-in","secure","security","account",
    "verify","verification","update","confirm","password",
    "wallet","billing","bank","pay","credential","suspend",
]

# Domain suffixes that take TWO labels to form the real root
# (google.co.uk -> root is "google.co.uk", not "co.uk").
# Not exhaustive, but covers the common cases.
MULTI_PART_TLDS = {
    "co.uk","org.uk","gov.uk","ac.uk","co.in","co.jp","co.kr",
    "com.au","net.au","org.au","com.br","com.mx","co.za",
    "com.sg","com.hk","co.nz",
}

# Digit / symbol leetspeak
LEET_MAP = str.maketrans({
    "0": "o", "1": "l", "!": "i", "3": "e",
    "4": "a", "5": "s", "7": "t", "$": "s", "@": "a",
})

# Multi-character visual tricks (checked as substring replacements,
# applied before the single-char leet map)
HOMOGLYPH_PAIRS = [
    ("rn", "m"), ("vv", "w"), ("cl", "d"), ("ii", "u"),
    ("nn", "m"), ("l1", "ll"),
]

# Cyrillic / Greek characters that render near-identically to Latin
# letters — classic IDN-homograph phishing trick.
UNICODE_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "і": "i", "ѕ": "s", "ԁ": "d",
    "ɡ": "g", "ⅼ": "l", "α": "a", "ο": "o", "ρ": "p",
})

def _has_non_ascii(s):
    return any(ord(ch) > 127 for ch in s)

def _visual_normalize(s):
    """Collapse leetspeak, multi-char visual tricks, and unicode
    homoglyphs down to a plain-ASCII 'what a human would read' form."""
    s = s.lower().translate(UNICODE_HOMOGLYPHS)
    for pair, repl in HOMOGLYPH_PAIRS:
        s = s.replace(pair, repl)
    return s.translate(LEET_MAP)

def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,
                         cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]

def _split_root(dom_lower):
    """Best-effort root-domain extraction, aware of common
    two-label TLDs like co.uk so 'google.co.uk' isn't split wrong."""
    labels = dom_lower.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in MULTI_PART_TLDS:
        return ".".join(labels[-3:]), labels[-3]
    if len(labels) >= 2:
        return ".".join(labels[-2:]), labels[-2]
    return dom_lower, dom_lower

def _decode_punycode_domain(dom_lower):
    """Decode any xn-- labels to their real unicode form so IDN
    homograph tricks (e.g. xn--ggle-0nda.com -> gòogle.com) can be
    compared against brand names, not just flagged blindly."""
    out_labels = []
    changed = False
    for label in dom_lower.split("."):
        if label.startswith("xn--"):
            try:
                out_labels.append(label[4:].encode("ascii").decode("punycode"))
                changed = True
                continue
            except Exception:
                pass
        out_labels.append(label)
    return ".".join(out_labels), changed

def brand_impersonation_check(raw_url, domain):
    """
    Returns a risk assessment dict:
      typosquat                  -> leetspeak/near-miss of a known brand
      homoglyph                  -> visual-trick or unicode/punycode lookalike
      brand_in_subdomain         -> brand present in URL but not the real root
      punycode                   -> domain uses IDN/punycode encoding (xn--)
      keyword_count               -> count of sensitive words in the URL
      keyword_stacked_subdomains  -> 2+ sensitive keywords each as their own
                                      subdomain label (e.g. bank.verify.account.*)
      credential_char_abuse       -> '@' or '\' present in the domain
                                      (userinfo/host-spoofing trick)
      matched_brand                -> which brand it resembles, if any
      risk_score                    -> 0.0-1.0 weighted combination of the above
    """
    flags = {
        "typosquat": False,
        "homoglyph": False,
        "brand_in_subdomain": False,
        "punycode": False,
        "keyword_count": 0,
        "keyword_stacked_subdomains": False,
        "credential_char_abuse": False,
        "matched_brand": None,
        "risk_score": 0.0,
    }

    dom_lower = domain.lower()

    # ── '@' / '\' host-spoofing check ───────────────────────
    # A literal '@' or '\' in the domain text is almost never
    # legitimate. Browsers treat everything before '@' as userinfo,
    # so "bankdetails\otp@evil.com" LOOKS like the host is
    # "bankdetails\otp" but the real host is "evil.com".
    if "@" in domain or "\\" in domain:
        flags["credential_char_abuse"] = True

    if "xn--" in dom_lower:
        flags["punycode"] = True
        decoded, _ = _decode_punycode_domain(dom_lower)
        dom_lower = decoded  # compare the real (decoded) text from here on

    root_domain, main_label = _split_root(dom_lower)

    # A domain is only "the real thing" if its actual root label is the
    # brand itself (any TLD: .com, .co.uk, .de, ...) — TLD doesn't matter.
    if main_label in BRANDS:
        # Even a "legit" root label doesn't excuse @ / \ abuse — that
        # means the "root" we parsed isn't the real browser-resolved host.
        if flags["credential_char_abuse"]:
            flags["risk_score"] = 0.55
        return flags  # legitimate root domain (e.g. google.co.uk) — skip entirely

    unicode_trick = _has_non_ascii(domain) or flags["punycode"]
    normalized_main = _visual_normalize(main_label)
    dom_normalized  = _visual_normalize(dom_lower)

    for brand in BRANDS:
        if main_label == brand:
            continue  # legitimate root domain under any TLD — never flag it

        # dist==0 catches exact tricks (g00gle/rnicrosoft -> google/microsoft);
        # dist 1-2 catches near-misses (googel, gogle, paypaI, etc.)
        dist = _levenshtein(normalized_main, brand)
        if dist <= 2 and abs(len(normalized_main) - len(brand)) <= 2:
            if dist == 0:
                # Perfect match only after visual normalization —
                # i.e. the raw text used a trick, not real letters.
                flags["homoglyph" if unicode_trick else "typosquat"] = True
            else:
                flags["typosquat"] = True
            flags["matched_brand"] = brand

        if ((brand in dom_lower or brand in dom_normalized)
                and main_label != brand):
            flags["brand_in_subdomain"] = True
            flags["matched_brand"] = flags["matched_brand"] or brand

    flags["keyword_count"] = sum(
        1 for kw in SENSITIVE_KEYWORDS if kw in raw_url.lower())

    # ── Subdomain-stacking check ─────────────────────────────
    # Flags URLs that stuff multiple trust keywords into their own
    # subdomain labels, e.g. bank.verify.account.evil-domain.com.
    # Real sites rarely chain "verify" and "account" as separate
    # subdomain labels — phishing sites do it because it reads as
    # reassuring to a skimming human eye.
    labels = dom_lower.split(".")
    subdomain_labels = labels[:-2] if len(labels) > 2 else []
    stacked_hits = sum(
        1 for label in subdomain_labels
        if label in SENSITIVE_KEYWORDS
        or any(kw in label for kw in SENSITIVE_KEYWORDS)
    )
    if stacked_hits >= 2:
        flags["keyword_stacked_subdomains"] = True

    # ── Weighted risk score ─────────────────────────────
    score = 0.0
    if flags["typosquat"]:
        score += 0.55
    if flags["homoglyph"]:
        score += 0.60
    if flags["brand_in_subdomain"]:
        score += 0.50
    if flags["punycode"] and flags["matched_brand"]:
        score += 0.35
    if flags["credential_char_abuse"]:
        score += 0.55
    if flags["keyword_stacked_subdomains"]:
        score += 0.45
    # Keyword score: cap raised 0.36 -> 0.65. The old cap made it
    # mathematically impossible for keyword density alone to ever
    # cross a 0.5 classification threshold, no matter how stuffed
    # the URL was.
    score += min(flags["keyword_count"] * 0.15, 0.65)
    flags["risk_score"] = min(score, 0.99)

    return flags

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
        print(f"[ML MODEL] {label}  "
              f"phish={prob[1]:.3f}  legit={prob[0]:.3f}")

        # ── Brand-impersonation risk blend ───────────────
        # Runs independently of the ML model, which has no
        # visibility into brand names, leetspeak, or homoglyphs.
        domain_for_check = urlparse(
            raw_url if raw_url.startswith(("http://", "https://"))
            else "http://" + raw_url
        ).netloc.split(":")[0]

        bi = brand_impersonation_check(raw_url, domain_for_check)
        prob = list(prob)

        if bi["risk_score"] > 0:
            # Blend: final phishing prob is whichever is higher of the
            # ML model's own estimate and the rule-based risk score,
            # rather than always slamming to a fixed 97%. This keeps
            # low-confidence signals (e.g. 2 keywords alone) graded,
            # while strong matches (typosquat/homoglyph/lookalike
            # domain/host-spoofing) still push the result decisively
            # to "Phishing".
            prob[1] = max(prob[1], bi["risk_score"])
            prob[0] = 1 - prob[1]
            pred = 1 if prob[1] >= 0.5 else pred

            reasons = []
            if bi["typosquat"]:
                reasons.append(f"typosquat of '{bi['matched_brand']}'")
            if bi["homoglyph"]:
                reasons.append(f"homoglyph/visual trick mimicking '{bi['matched_brand']}'")
            if bi["brand_in_subdomain"]:
                reasons.append(f"'{bi['matched_brand']}' used outside its real domain")
            if bi["punycode"]:
                reasons.append("punycode/IDN-encoded domain")
            if bi["credential_char_abuse"]:
                reasons.append("'@' or '\\' abuse in domain (host-spoofing attempt)")
            if bi["keyword_stacked_subdomains"]:
                reasons.append("sensitive keywords stacked as separate subdomains")
            if bi["keyword_count"] >= 2:
                reasons.append(f"{bi['keyword_count']} sensitive keywords")
            print(f"[RISK LAYER] score={bi['risk_score']:.2f} — " + "; ".join(reasons))

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
