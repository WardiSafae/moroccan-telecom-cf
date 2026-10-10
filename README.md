# Moroccan Telecom Package Recommender

Implicit-feedback collaborative filtering system that recommends the
next best mobile package to Moroccan prepaid subscribers — built as a
portfolio project targeting Data Science / ML roles in Morocco's
telecom and banking sectors.

**🔗 Live demo:** [https://moroccan-telecom-cf-bdsi.streamlit.app/]
**📓 Notebook:** [https://www.kaggle.com/code/wardisafae/moroccan-telecom-cf-recommender]

---

## Business context

Morocco's three main operators (Maroc Telecom, Orange Maroc, inwi) serve
millions of prepaid subscribers. Most stay on low-value plans because
they don't know which package fits their usage. Manual segmentation is
too coarse to personalize at scale.

This project builds a **collaborative filtering recommender** that
suggests the next best package for each subscriber, based on the
behavior of similar users.

**Why collaborative filtering:** subscriber preferences are *implicit*
(no star ratings — just recharges and usage), and CF handles implicit
feedback natively.

---

## Dataset

Real subscriber data is not publicly available, so the project uses a
**reproducible synthetic generator** that mimics Morocco-specific
patterns:

- **5,000 subscribers** across all 12 Moroccan regions (weighted by
  population)
- **200 packages** spanning voice, data, combo, and international
  (diaspora) offerings across low/mid/high segments
- **~57,000 interactions** with Morocco-relevant signals:
  - **Ramadan seasonality** (35% of users get a 2× boost)
  - **Diaspora calling patterns** (Europe, Maghreb, World tiers)
  - **Regional distribution** matching Morocco's population density

**Sparsity:** 94.3% — realistic for telecom implicit-feedback datasets.

---

## Methodology

### Interaction model
Each subscriber interacts with 3–20 distinct packages. Implicit strength is
computed as:
```
confidence = recharges × 1.0 + usage_hours × 0.1
```
This produces a sparse user × item matrix used by both models.

### Models compared

| Model | Approach |
|---|---|
| **Item-Item CF** | Cosine similarity on item vectors weighted by user's interaction strength |
| **ALS Matrix Factorization** | `implicit` library, 128 latent factors, α=80, reg=0.01 |

### Train/test split
Per-user **time-based split** (80/20): the last 20% of each user's
interactions form the test set — mirrors how production systems
evaluate "predict the next action".

### Evaluation metrics
Ranking metrics suited to implicit feedback:
- Precision@K, Recall@K
- **NDCG@K** (position-aware)
- MAP@K
- **Catalog coverage** (detects popularity bias)

### Hyperparameter search
27-configuration grid over `factors ∈ {64,128,256}`,
`alpha ∈ {20,40,80}`, `reg ∈ {0.01,0.05,0.1}`. Best config:
`factors=256, alpha=20, reg=0.1` → **NDCG@10 = 0.427** on 500 users
(+58% vs untuned baseline).

---

## Results

| Metric | Item-Item CF | ALS (MF) |
|---|---|---|
| P@5 | **0.2368** | 0.2184 |
| P@10 | **0.1564** | 0.1464 |
| R@5 | **0.4328** | 0.3967 |
| R@10 | **0.5565** | 0.5277 |
| NDCG@5 | **0.4101** | 0.3721 |
| NDCG@10 | **0.4663** | 0.4298 |
| MAP@5 | **0.3350** | 0.2953 |
| MAP@10 | **0.3704** | 0.3298 |
| Coverage@10 | 0.7600 | **0.9550** |

### Takeaways

1. **Item-Item CF leads on ranking accuracy** (NDCG@10 = 0.466 vs 0.430) —
   its local co-occurrence signal is a strong fit for a 200-item catalog.
2. **ALS achieves 25% broader catalog coverage** (95.5% vs 76%) — it
   surfaces long-tail packages that Item-Item CF never recommends.
3. **Business implication:** for a telecom operator prioritizing diversity
   and long-tail discovery (upsell to underserved segments), ALS's coverage
   advantage may outweigh its ~8% accuracy gap. For pure click-through
   optimization, Item-Item CF is preferable.
4. **Hyperparameter search mattered:** a 27-config grid improved ALS
   NDCG@10 by +6% and coverage by +2.5 pp over the untuned baseline.

This is the classic **accuracy vs. diversity trade-off** in recommender
systems, and choosing between the two should be driven by the operator's
business objective.

---

## Cold-start handling

Two fallback paths for subscribers with no interaction history:
1. **Unknown user ID** → segment-based popularity ranking
2. **Known user with no training interactions** → same popularity
   fallback

---

## Deployment

Interactive **Streamlit** demo with three views:
- **Recommend for user** — enter subscriber ID → top-N packages
- **Similar packages** — item-item similarity explorer
- **Model metrics** — evaluation dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
moroccan-telecom-cf/
├── app.py                      # Streamlit demo
├── requirements.txt
├── README.md
├── notebooks/
│   └── morocco_telecom_cf.ipynb
└── artifacts/
    ├── als_model.pkl
    ├── item_similarity.pkl
    ├── train_user_item.npz
    ├── encoders.pkl
    ├── known_users.pkl
    └── packages.csv
```

## Tech stack
Python · Pandas · NumPy · SciPy (sparse) · implicit ·
Scikit-learn · Streamlit · Matplotlib · Seaborn
