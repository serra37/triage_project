Triage Project
LangGraph tabanli tibbi triyaj asistani. Uygulama kullanicinin sikayetini CLI uzerinden alir, eksik bilgi varsa netlestirici soru sorar, ardindan ChromaDB + BM25 + Cohere rerank destekli RAG akisi ile genel bir yonlendirme uretir.

Not: Bu proje tibbi tani koymak icin degil, bilgilendirme ve yonlendirme amaciyla gelistirilmektedir.

Kurulum
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
.env.example dosyasini .env olarak kopyalayip gerekli anahtarlari doldurun:

OPENAI_API_KEY=...
COHERE_API_KEY=...
Veri Tabanini Hazirlama
RAG aramasinin calisabilmesi icin once ChromaDB verisini olusturun:

python data_pipeline/ingest.py
Varsayilan olarak her veri setinden 500 kayit islenir. Bu islem OpenAI API kullanir ve zaman alabilir.

Calistirma
python main.py
CLI acildiktan sonra sikayetinizi yazin. Cikmak icin q girin.

Proje Yapisi
main.py: Komut satiri arayuzu ve sohbet dongusu.
graph.py: LangGraph akis tanimi.
agents/nodes.py: Supervisor, intent, RAG ve klinik karar dugumleri.
agents/state.py: Graf state modeli.
database/chroma_client.py: Chroma, BM25, RRF ve Cohere rerank arama yardimcilari.
data_pipeline/ingest.py: Veri setlerini indirip chunk'layarak ChromaDB'ye ekler.
Gelistirme Notlari
Kisa vadede faydali olacak iyilestirmeler:

Web arayuzu eklemek.
Unit test ve smoke test eklemek.
Promptlari daha guvenli, tutarli ve JSON hatalarina dayanikli hale getirmek.
Triage kararini aciliyet seviyesi ve bolum onerisi gibi yapisal alanlara ayirmak.
RAG veri isleme adimini daha ucuz ve tekrar calistirilabilir hale getirmek.
