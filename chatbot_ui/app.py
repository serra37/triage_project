"""
Tıbbi Triyaj Asistanı – Streamlit Chatbot UI
Demo modu: Backend bağlantısı olmadan kural tabanlı yanıt üretir.
Sadece chatbot_ui/app.py değiştirilmiştir.
"""

import streamlit as st
import uuid
import re
import time
from datetime import datetime

# ── Sayfa yapılandırması ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Triyaj Asistanı",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Risk seviyesi tanımları ────────────────────────────────────────────────────
RISK_LEVELS = {
    "ACİL": {
        "color": "#DC2626",       # kırmızı
        "bg": "#FEF2F2",
        "border": "#FECACA",
        "icon": "🚨",
        "yonlendirme": "Derhal 112'yi arayın veya en yakın acil servise başvurun.",
        "guvenlik": "Bu değerlendirme tanı niteliği taşımaz. Acil durumlarda vakit kaybetmeyin.",
    },
    "YÜKSEK": {
        "color": "#EA580C",       # turuncu-kırmızı
        "bg": "#FFF7ED",
        "border": "#FED7AA",
        "icon": "⚠️",
        "yonlendirme": "En kısa sürede bir acil servise veya nöbetçi hekime başvurun.",
        "guvenlik": "Semptomlarınız hızla kötüleşirse 112'yi arayın.",
    },
    "ORTA": {
        "color": "#D97706",       # sarı-turuncu
        "bg": "#FFFBEB",
        "border": "#FDE68A",
        "icon": "🔶",
        "yonlendirme": "Bugün bir sağlık kuruluşuna başvurmanız önerilir.",
        "guvenlik": "Şikayetleriniz artarsa doğrudan acil servise gidin.",
    },
    "DÜŞÜK": {
        "color": "#16A34A",       # yeşil
        "bg": "#F0FDF4",
        "border": "#BBF7D0",
        "icon": "✅",
        "yonlendirme": "Yakın zamanda bir aile hekimine randevu alabilirsiniz.",
        "guvenlik": "Yeni belirtiler gelişirse tekrar değerlendirme yaptırın.",
    },
}

# ── Demo: Kural tabanlı risk analizi ──────────────────────────────────────────
ACIL_KEYWORDS = [
    "göğüs ağrısı", "nefes alamıyorum", "sol kol", "felç", "bilinç",
    "bayıldım", "bayılıyorum", "inme", "kalp", "şiddetli ağrı", "kan kusuyorum",
    "kaza", "yanma", "yakıyor", "elektrik çarptı", "boğulma",
]
YUKSEK_KEYWORDS = [
    "yüksek ateş", "40 derece", "39 derece", "şiddetli baş ağrısı",
    "kusma", "karın ağrısı", "solunum güçlüğü", "çarpıntı",
    "idrar yapamıyorum", "ani görme kaybı",
]
ORTA_KEYWORDS = [
    "ateş", "38", "boğaz ağrısı", "öksürük", "baş ağrısı", "mide",
    "ishal", "bulantı", "eklem ağrısı", "halsizlik", "grip",
]


def demo_risk_analizi(metin: str) -> dict:
    """Kullanıcı metninden kural tabanlı risk seviyesi çıkarır."""
    m = metin.lower()
    if any(k in m for k in ACIL_KEYWORDS):
        seviye = "ACİL"
        aciklama = (
            "Mesajınızda hayatı tehdit edebilecek belirtiler tespit edildi. "
            "Lütfen hemen yardım alın."
        )
        yanit = (
            "Belirttiğiniz semptomlar **acil müdahale** gerektirebilir. "
            "Sistem bir ön değerlendirme yapmıştır, ancak bu kesin tanı değildir."
        )
    elif any(k in m for k in YUKSEK_KEYWORDS):
        seviye = "YÜKSEK"
        aciklama = "Yüksek risk taşıyan semptomlar tespit edildi. En kısa sürede sağlık kuruluşuna başvurun."
        yanit = (
            "Semptomlarınız **yüksek risk** kategorisinde değerlendirildi. "
            "Mümkün olan en kısa sürede bir hekime görünmeniz önerilir."
        )
    elif any(k in m for k in ORTA_KEYWORDS):
        seviye = "ORTA"
        aciklama = "Orta düzey belirtiler mevcut. Bugün bir sağlık kuruluşuna başvurun."
        yanit = (
            "Şikayetleriniz **orta risk** seviyesinde görünüyor. "
            "Bugün içinde bir sağlık profesyoneliyle görüşmeniz tavsiye edilir."
        )
    else:
        seviye = "DÜŞÜK"
        aciklama = "Acil bir belirti tespit edilmedi. Şikayetlerinizi takip edin."
        yanit = (
            "Belirttiğiniz şikayetler şu an için **düşük risk** olarak değerlendirildi. "
            "Belirtilerinizin seyrine dikkat edin; kötüleşirse tekrar başvurun."
        )
    return {
        "reply": yanit,
        "triage_level": seviye,
        "aciklama": aciklama,
        "final_decision": True,
    }


# ── Session State başlatma ─────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],          # {"role", "content", "triage_level", "aciklama"}
        "show_welcome": True,
        "scan_history": [        # Göstermelik geçmiş taramalar
            {"label": "Göğüs ağrısı – ACİL", "ago": "2 saat önce"},
            {"label": "Ateş & boğaz – ORTA", "ago": "Dün"},
            {"label": "Baş ağrısı – DÜŞÜK", "ago": "3 gün önce"},
        ],
        "pending_prompt": None,  # Örnek kart tıklaması
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def reset_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.show_welcome = True
    st.session_state.pending_prompt = None


# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Genel arka plan */
.stApp { background: #F1F5F9; }

/* Üst çubuğu gizle */
#MainMenu, header, footer { visibility: hidden; }

/* Sidebar stil */
section[data-testid="stSidebar"] > div:first-child {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
    padding: 0 !important;
}

/* Sohbet alanı arka planı */
.main .block-container {
    max-width: 860px;
    padding: 1rem 1.5rem 6rem 1.5rem;
    background: transparent;
}

/* Kullanıcı balonu */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 8px 0;
}
.msg-user .bubble {
    background: #2563EB;
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.93rem;
    line-height: 1.55;
    box-shadow: 0 1px 4px rgba(37,99,235,0.25);
}

/* Asistan balonu */
.msg-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 8px 0;
}
.msg-assistant .bubble {
    background: #FFFFFF;
    color: #1E293B;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    max-width: 72%;
    font-size: 0.93rem;
    line-height: 1.55;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Risk sonuç kartı */
.risk-result-card {
    border-radius: 12px;
    padding: 20px 24px;
    margin: 14px 0;
    border: 1px solid;
}
.risk-result-card .risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 12px;
}
.risk-result-card .risk-section {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 6px;
    line-height: 1.5;
}
.risk-result-card .risk-label {
    font-weight: 600;
    color: #1E293B;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 10px;
}

/* Hoş geldin ekranı */
.welcome-area {
    text-align: center;
    padding: 48px 24px 32px;
}
.welcome-icon {
    font-size: 4rem;
    margin-bottom: 16px;
}
.welcome-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 12px;
}
.welcome-desc {
    font-size: 0.97rem;
    color: #64748B;
    max-width: 520px;
    margin: 0 auto 28px;
    line-height: 1.65;
}

/* Örnek prompt kartları */
.prompt-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 18px;
    cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s;
    text-align: left;
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.5;
    margin-bottom: 10px;
}
.prompt-card:hover {
    border-color: #2563EB;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.10);
}
.prompt-card .pc-icon { font-size: 1.1rem; margin-right: 6px; }

/* Güvenlik uyarısı */
.safety-bar {
    position: fixed;
    bottom: 70px;
    left: 0; right: 0;
    background: #FFFBEB;
    border-top: 1px solid #FDE68A;
    color: #92400E;
    font-size: 0.78rem;
    text-align: center;
    padding: 6px 16px;
    z-index: 999;
}

/* Sidebar öğeleri */
.sb-header {
    padding: 20px 20px 16px;
    border-bottom: 1px solid #F1F5F9;
}
.sb-logo { font-size: 1.1rem; font-weight: 700; color: #1E293B; }
.sb-sub  { font-size: 0.78rem; color: #94A3B8; margin-top: 2px; }

.sb-section-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #94A3B8;
    padding: 16px 20px 6px;
}

.hist-item {
    padding: 9px 20px;
    border-radius: 0;
    font-size: 0.85rem;
    color: #475569;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: background 0.15s;
}
.hist-item:hover { background: #F8FAFC; }
.hist-item .hist-time { font-size: 0.75rem; color: #94A3B8; }

.sb-user-card {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: 14px 20px;
    border-top: 1px solid #F1F5F9;
    display: flex;
    align-items: center;
    gap: 12px;
    background: #FFFFFF;
}
.sb-avatar {
    width: 34px; height: 34px;
    background: #EFF6FF;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.sb-user-name  { font-size: 0.85rem; font-weight: 600; color: #1E293B; }
.sb-user-title { font-size: 0.75rem; color: #94A3B8; }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo & başlık
    st.markdown("""
    <div class="sb-header">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.6rem;">🏥</span>
            <div>
                <div class="sb-logo">Triyaj Asistanı</div>
                <div class="sb-sub">Yapay Zeka Destekli</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Yeni Tıbbi Kayıt butonu
    st.markdown("<div style='padding:12px 16px;'>", unsafe_allow_html=True)
    if st.button("＋  Yeni Tıbbi Kayıt", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Geçmiş taramalar
    st.markdown('<div class="sb-section-title">Geçmiş Taramalar</div>', unsafe_allow_html=True)
    for item in st.session_state.scan_history:
        st.markdown(f"""
        <div class="hist-item">
            <div>💬 {item['label']}</div>
            <div class="hist-time">{item['ago']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Divider
    st.markdown("<hr style='margin:16px 0;border:none;border-top:1px solid #F1F5F9;'>", unsafe_allow_html=True)

    # Risk seviyesi kılavuzu
    st.markdown('<div class="sb-section-title">Risk Seviyeleri</div>', unsafe_allow_html=True)
    guide = [
        ("ACİL",   "#DC2626", "Hayatı tehdit eden durum"),
        ("YÜKSEK", "#EA580C", "Acil sağlık gerektirir"),
        ("ORTA",   "#D97706", "Bugün başvuru önerilir"),
        ("DÜŞÜK",  "#16A34A", "Rutin takip yeterli"),
    ]
    for lbl, clr, desc in guide:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:6px 20px;">
            <div style="width:10px;height:10px;border-radius:50%;background:{clr};flex-shrink:0;"></div>
            <div>
                <span style="font-size:0.82rem;font-weight:600;color:{clr};">{lbl}</span>
                <span style="font-size:0.78rem;color:#94A3B8;margin-left:6px;">{desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Kullanıcı kartı (sayfa altına sabitlenmiş görünüm)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-user-card">
        <div class="sb-avatar">👨‍⚕️</div>
        <div>
            <div class="sb-user-name">Görevli Hekim</div>
            <div class="sb-user-title">Poliklinik Girişi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── ANA ALAN ───────────────────────────────────────────────────────────────────

def render_risk_card(level: str, aciklama: str) -> None:
    """Risk sonucu kartı."""
    r = RISK_LEVELS.get(level, RISK_LEVELS["DÜŞÜK"])
    st.markdown(f"""
    <div class="risk-result-card"
         style="background:{r['bg']};border-color:{r['border']};">
        <div class="risk-badge" style="color:{r['color']};">
            {r['icon']} Ön Risk Seviyesi: {level}
        </div>
        <div class="risk-label">Açıklama</div>
        <div class="risk-section">{aciklama}</div>
        <div class="risk-label">Önerilen Yönlendirme</div>
        <div class="risk-section">{r['yonlendirme']}</div>
        <div class="risk-label">Güvenlik Notu</div>
        <div class="risk-section">{r['guvenlik']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_chat_history() -> None:
    """Tüm mesajları baloncuklar ile gösterir."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-assistant">
                <div class="bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
            if msg.get("triage_level"):
                render_risk_card(msg["triage_level"], msg.get("aciklama", ""))


# ── HOŞ GELDİN EKRANI ─────────────────────────────────────────────────────────
PROMPTS = [
    ("🫀", "Şiddetli göğüs ağrım var ve sol koluma vuruyor. Tansiyonum çok yüksek."),
    ("🌡️", "Dünden beri 38.5 ateşim var, boğazım ağrıyor ve yutkunamıyorum."),
]

if st.session_state.show_welcome and not st.session_state.messages:
    st.markdown("""
    <div class="welcome-area">
        <div class="welcome-icon">🛡️</div>
        <div class="welcome-title">Dijital Triyaj Asistanına Hoş Geldiniz</div>
        <div class="welcome-desc">
            Lütfen hastanın şikayetlerini, yaşını ve varsa kronik rahatsızlıklarını
            detaylı şekilde yazın. Yapay zeka sistemimiz aciliyet durumunu ön
            değerlendirecektir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Örnek prompt kartları – Streamlit butonları olarak
    st.markdown("<div style='max-width:580px;margin:0 auto;'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            f"{PROMPTS[0][0]}  {PROMPTS[0][1][:55]}…",
            use_container_width=True,
            key="prompt_0",
        ):
            st.session_state.pending_prompt = PROMPTS[0][1]
            st.session_state.show_welcome = False
            st.rerun()
    with col2:
        if st.button(
            f"{PROMPTS[1][0]}  {PROMPTS[1][1][:55]}…",
            use_container_width=True,
            key="prompt_1",
        ):
            st.session_state.pending_prompt = PROMPTS[1][1]
            st.session_state.show_welcome = False
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Geçmiş mesajları göster
    render_chat_history()


# ── ÖRNEK PROMPT OTOMATİK GÖNDERİM ───────────────────────────────────────────
if st.session_state.pending_prompt:
    user_text = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    # Kullanıcı mesajını kaydet
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "triage_level": None,
        "aciklama": "",
    })

    # Demo analiz
    with st.spinner("Sistem değerlendiriyor…"):
        time.sleep(0.8)
        result = demo_risk_analizi(user_text)

    # Asistan yanıtını kaydet
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["reply"],
        "triage_level": result["triage_level"],
        "aciklama": result["aciklama"],
    })

    # Geçmiş taramalar listesine ekle
    ts = datetime.now().strftime("%H:%M")
    short = user_text[:25] + "…"
    st.session_state.scan_history.insert(0, {
        "label": f"{short} – {result['triage_level']}",
        "ago": f"Bugün {ts}",
    })
    st.session_state.show_welcome = False
    st.rerun()


# ── CHAT INPUT ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Hastanın şikayetini yazın...")

if user_input:
    st.session_state.show_welcome = False

    # Kullanıcı mesajını kaydet
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "triage_level": None,
        "aciklama": "",
    })

    # Demo analiz
    with st.spinner("Sistem değerlendiriyor…"):
        time.sleep(0.8)
        result = demo_risk_analizi(user_input)

    # Asistan yanıtını kaydet
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["reply"],
        "triage_level": result["triage_level"],
        "aciklama": result["aciklama"],
    })

    # Geçmiş taramalar listesine ekle
    ts = datetime.now().strftime("%H:%M")
    short = user_input[:25] + "…"
    st.session_state.scan_history.insert(0, {
        "label": f"{short} – {result['triage_level']}",
        "ago": f"Bugün {ts}",
    })
    st.rerun()


# ── GÜVENLIK UYARISI (sabit alt bar) ─────────────────────────────────────────
st.markdown("""
<div class="safety-bar">
    ⚠️ <strong>Uyarı:</strong> Bu sistem tıbbi tavsiye niteliği taşımaz ve
    sadece ön değerlendirme amaçlıdır. Acil durumlarda <strong>112</strong>'yi arayın.
</div>
""", unsafe_allow_html=True)