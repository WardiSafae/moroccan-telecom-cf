import streamlit as st
import pickle, os
import numpy as np
import pandas as pd
from scipy import sparse

st.set_page_config(
    page_title="Morocco Telecom Recommender",
    page_icon="📱",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    with open("artifacts/als_model.pkl", "rb") as f:
        als_model = pickle.load(f)
    with open("artifacts/encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    with open("artifacts/item_similarity.pkl", "rb") as f:
        item_sim = pickle.load(f)
    with open("artifacts/known_users.pkl", "rb") as f:
        known_users = pickle.load(f)

    train_user_item = sparse.load_npz("artifacts/train_user_item.npz")
    packages = pd.read_csv("artifacts/packages.csv")

    return {
        "als": als_model,
        "user_enc": enc["user_enc"],
        "item_enc": enc["item_enc"],
        "item_sim": item_sim,
        "train_user_item": train_user_item,
        "packages": packages,
        "known_users": known_users,
    }


A = load_artifacts()
packages = A["packages"]
als_model = A["als"]
user_enc = A["user_enc"]
item_enc = A["item_enc"]
train_user_item = A["train_user_item"]
known_users = A["known_users"]

st.title("📱 Morocco Telecom Package Recommender")
st.caption(
    "Collaborative Filtering · ALS Matrix Factorization · "
    "Synthetic Maroc Telecom-style dataset (5,000 subscribers × 200 packages)"
)

tab1, tab2, tab3 = st.tabs(
    ["🎯 Recommend for user", "🔍 Similar packages", "📊 Model metrics"]
)

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Get top-N package recommendations")
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.number_input(
            "Subscriber ID",
            min_value=0,
            max_value=max(known_users),
            value=42,
            step=1,
        )
    with col2:
        top_n = st.slider("Top N", 3, 10, 5)

    if st.button("Recommend", type="primary"):
        cold_start = False
        if user_id not in known_users:
            cold_start = True
        else:
            u_idx = user_enc.transform([user_id])[0]
            if train_user_item[u_idx].nnz == 0:
                cold_start = True

        if cold_start:
            st.warning(
                "Cold-start user — no interaction history. "
                "Showing segment-popularity fallback."
            )
            recs_df = packages.sample(top_n, random_state=42)[
                ["package_id", "name", "type", "price_mad", "segment"]
            ]
        else:
            u_idx = user_enc.transform([user_id])[0]
            ids, scores = als_model.recommend(
                u_idx,
                train_user_item[u_idx],
                N=top_n,
                filter_already_liked_items=True,
            )
            pkg_ids = item_enc.inverse_transform(ids)
            recs_df = packages[packages.package_id.isin(pkg_ids)].copy()
            recs_df["score"] = scores
            recs_df = recs_df.sort_values("score", ascending=False)
            recs_df = recs_df[
                ["package_id", "name", "type", "price_mad", "segment", "score"]
            ]

        st.dataframe(recs_df, use_container_width=True)

        # Show user's past interactions
        if not cold_start:
            st.markdown("**This user's interactions in the training set:**")
            row = train_user_item[u_idx].toarray().ravel()
            past = np.where(row > 0)[0]
            if len(past):
                past_ids = item_enc.inverse_transform(past)
                st.dataframe(
                    packages[packages.package_id.isin(past_ids)][
                        ["package_id", "name", "type", "price_mad"]
                    ],
                    use_container_width=True,
                )

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Find packages similar to a given one")
    st.caption(
        "Item-item cosine similarity — powers the Item-Item CF baseline model."
    )
    pkg_choice = st.selectbox(
        "Choose a package",
        options=packages["package_id"].tolist(),
        format_func=lambda x: (
            f"{x} — {packages.loc[packages.package_id == x, 'name'].values[0]}"
        ),
    )
    item_sim = A["item_sim"]
    sims = item_sim[pkg_choice]
    top_sim_idx = np.argsort(sims)[::-1][:6]
    sim_df = packages.iloc[top_sim_idx][
        ["package_id", "name", "type", "price_mad", "segment"]
    ].copy()
    sim_df["similarity"] = sims[top_sim_idx]
    st.dataframe(sim_df, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Model performance (offline evaluation)")
    st.caption(
        "Evaluated on a per-user time-based 80/20 split. "
        "ALS tuned via 27-config grid search."
    )

    metrics = pd.DataFrame(
        {
            "Item-Item CF": [0.2368, 0.5565, 0.4663, 0.3704, 0.7600],
            "ALS (MF)": [0.2184, 0.5277, 0.4298, 0.3298, 0.9550],
        },
        index=["P@5", "R@10", "NDCG@10", "MAP@10", "Coverage@10"],
    )
    st.dataframe(metrics, use_container_width=True)
    st.bar_chart(metrics.T)

    st.markdown(
        """
        **Takeaways**

        - **Item-Item CF leads on ranking accuracy** (NDCG@10 = 0.466 vs 0.430).
        - **ALS achieves 95.5% catalog coverage** vs 76% for Item-Item CF — it
          surfaces long-tail packages that Item-Item CF never recommends.
        - **Business implication:** for diversity and upsell discovery, ALS's
          coverage advantage may outweigh its ~8% accuracy gap.
        """
    )

st.divider()
st.caption(
    "Built by WardiSafae · "
    "[GitHub](https://github.com/WardiSafae/moroccan-telecom-cf) · "
    "Portfolio project"
)
