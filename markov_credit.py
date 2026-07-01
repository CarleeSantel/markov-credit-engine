import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.linalg import expm

st.set_page_config(page_title="Markov Chain Credit Risk Engine", layout="wide")
st.title("Markov Chain Credit Risk Engine")
st.caption(
    "MLE-parameterized rating transition matrix · Chapman-Kolmogorov · "
    "Absorbing state default dynamics · Portfolio expected loss · Stress testing"
)

# ── RATING STATES ─────────────────────────────────────────────────────────────
STATES = ["AAA","AA","A","BBB","BB","B","CCC","D"]
n_states = len(STATES)
D_IDX = STATES.index("D")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Transition Matrix", "Multi-Period Dynamics", "Portfolio Expected Loss", "Stress Testing"]
)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("Parameterization")
param_mode = st.sidebar.radio(
    "Matrix Source",
    ["Historical S&P Averages (1981–2023)", "Custom Upload / Manual Edit"],
)
st.sidebar.divider()
recovery = st.sidebar.slider("Recovery Rate (LGD = 1 - RR)", 0.0, 1.0, 0.40, step=0.05,
                              help="Recovery rate on defaulted bonds. LGD = 1 - recovery.")
st.sidebar.divider()
st.sidebar.subheader("Portfolio (for Tab 3)")

# ── HISTORICAL TRANSITION MATRIX ───────────────────────────────────────────────
# Source: S&P Global Annual Default Study 2023 (publicly available, academic standard)
# These are average 1-year transition probabilities across the 1981-2023 study period.
# D is absorbing (row sums to 1 with entire weight on D).
P_historical = np.array([
    # AAA     AA      A       BBB     BB      B       CCC     D
    [0.9168, 0.0715, 0.0083, 0.0025, 0.0006, 0.0001, 0.0001, 0.0001],  # AAA
    [0.0062, 0.9089, 0.0762, 0.0059, 0.0019, 0.0006, 0.0001, 0.0002],  # AA
    [0.0006, 0.0213, 0.9129, 0.0564, 0.0062, 0.0018, 0.0003, 0.0005],  # A
    [0.0004, 0.0023, 0.0429, 0.8826, 0.0528, 0.0143, 0.0019, 0.0028],  # BBB
    [0.0003, 0.0008, 0.0046, 0.0621, 0.8141, 0.0895, 0.0162, 0.0124],  # BB
    [0.0001, 0.0008, 0.0040, 0.0178, 0.0662, 0.8198, 0.0571, 0.0342],  # B
    [0.0000, 0.0005, 0.0017, 0.0062, 0.0173, 0.1048, 0.5937, 0.2758],  # CCC
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],  # D (absorbing)
])

# Ensure rows sum to 1
P_historical = P_historical / P_historical.sum(axis=1, keepdims=True)

if param_mode == "Historical S&P Averages (1981–2023)":
    P = P_historical.copy()
    st.sidebar.info("Using S&P Global 1-year average transition matrix (1981–2023 study).")
else:
    st.sidebar.markdown("**Custom Transition Matrix**")
    st.sidebar.caption("Edit the matrix in Tab 1. Rows must sum to 1.0.")
    P = P_historical.copy()  # editable in tab1

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRANSITION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("1-Year Rating Transition Matrix")
    st.markdown(
        "Each row is a starting rating; each cell is the probability of transitioning "
        "to that column rating in one year. Row sums = 1.0. **D (Default) is absorbing** — "
        "once a firm defaults, it stays in default."
    )

    if param_mode == "Custom Upload / Manual Edit":
        st.info("Paste your transition counts below — the MLE estimator will normalize rows.")
        raw_input = st.text_area(
            "Transition Count Matrix (8×8, space-separated, rows=AAA→CCC→D)",
            value="\n".join([" ".join([f"{int(P_historical[i,j]*1000):4d}" for j in range(n_states)])
                             for i in range(n_states)]),
            height=200
        )
        try:
            rows_in = [[float(x) for x in row.split()] for row in raw_input.strip().split("\n")]
            counts  = np.array(rows_in)
            if counts.shape == (n_states, n_states):
                row_sums = counts.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                P = counts / row_sums
                P[D_IDX] = 0
                P[D_IDX, D_IDX] = 1.0
                st.success("MLE matrix estimated from counts.")
            else:
                st.error(f"Expected {n_states}×{n_states} matrix, got {counts.shape}.")
                P = P_historical.copy()
        except Exception as e:
            st.error(f"Parse error: {e}")
            P = P_historical.copy()

    # Display as styled DataFrame
    df_P = pd.DataFrame(P, index=STATES, columns=STATES)
    st.dataframe(
        df_P.style.background_gradient(cmap="Blues", axis=1)
                  .format("{:.4f}"),
        use_container_width=True
    )
    st.caption("Source: S&P Global Annual Default Study 2023 — average 1-year corporate transition rates.")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Heatmap")
        fig = px.imshow(P, x=STATES, y=STATES, color_continuous_scale="Blues",
                         text_auto=".3f", labels=dict(x="To",y="From",color="Prob"))
        fig.update_layout(margin=dict(t=20,b=20), title="Transition Probability Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Default Probability (1-Year)")
        d_probs = P[:, D_IDX]
        fig2 = go.Figure(go.Bar(
            x=STATES[:-1], y=d_probs[:-1],
            text=[f"{p:.3%}" for p in d_probs[:-1]],
            textposition="outside",
            marker_color=["#2ecc71","#27ae60","#f1c40f","#e67e22","#e74c3c","#c0392b","#7b241c"]
        ))
        fig2.update_layout(title="1-Year Default Probability by Rating",
                            yaxis_tickformat=".2%", margin=dict(t=40,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Steady-State Distribution")
    st.caption("Where the system converges if left to evolve forever under this matrix.")
    P_100 = np.linalg.matrix_power(P, 100)
    steady = P_100[0]
    fig3 = px.bar(x=STATES, y=steady, labels={"x":"Rating","y":"Probability"},
                   title="Steady-State Distribution",
                   color=STATES, color_discrete_map={
                       "AAA":"#2ecc71","AA":"#27ae60","A":"#f1c40f","BBB":"#e67e22",
                       "BB":"#e74c3c","B":"#c0392b","CCC":"#7b241c","D":"#1a1a1a"})
    fig3.update_layout(showlegend=False, yaxis_tickformat=".1%", margin=dict(t=40,b=20))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Note: since D is absorbing, all probability mass eventually flows to Default under the stationary matrix.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MULTI-PERIOD DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Multi-Period Transition Dynamics — Chapman-Kolmogorov")
    st.markdown(
        r"The n-year transition matrix is $P^n$ — the matrix product of the annual matrix with itself n times. "
        r"This follows from the **Chapman-Kolmogorov equation**: "
        r"$P(s,u) = P(s,t) \cdot P(t,u)$ for $s < t < u$. "
        "It means future state probabilities compound through the Markov chain — no memory beyond the current rating."
    )

    start_rating = st.selectbox("Starting Rating", STATES[:-1], index=3)  # default BBB
    horizon_yrs  = st.slider("Horizon (years)", 1, 20, 10)

    # Compute cumulative default probability and rating distribution over time
    s_idx   = STATES.index(start_rating)
    init    = np.zeros(n_states); init[s_idx] = 1.0

    yearly_dists = [init.copy()]
    for yr in range(1, horizon_yrs+1):
        P_n = np.linalg.matrix_power(P, yr)
        yearly_dists.append(P_n[s_idx])

    yearly_dists = np.array(yearly_dists)
    cum_default  = yearly_dists[:, D_IDX]

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader(f"Cumulative Default Probability — Starting {start_rating}")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=list(range(horizon_yrs+1)), y=cum_default,
                                   mode="lines+markers",
                                   line=dict(color="#e74c3c",width=2.5),
                                   fill="tozeroy", fillcolor="rgba(231,76,60,0.15)"))
        fig4.update_layout(xaxis_title="Years", yaxis_title="P(Default)",
                            yaxis_tickformat=".2%", margin=dict(t=20,b=20))
        st.plotly_chart(fig4, use_container_width=True)

        # Hazard rate
        hazard = np.diff(cum_default) / (1 - cum_default[:-1] + 1e-10)
        fig4b = go.Figure()
        fig4b.add_trace(go.Bar(x=list(range(1,horizon_yrs+1)), y=hazard,
                                marker_color="#F58518",
                                text=[f"{h:.3%}" for h in hazard],
                                textposition="outside"))
        fig4b.update_layout(title="Annual Marginal Default Hazard Rate",
                             xaxis_title="Year", yaxis_title="Hazard Rate",
                             yaxis_tickformat=".2%", margin=dict(t=40,b=20))
        st.plotly_chart(fig4b, use_container_width=True)

    with col_d:
        st.subheader("Rating Distribution Over Time")
        fig5 = go.Figure()
        colors_rating = ["#2ecc71","#27ae60","#f1c40f","#e67e22","#e74c3c","#c0392b","#7b241c","#1a1a1a"]
        for j,state in enumerate(STATES):
            fig5.add_trace(go.Scatter(
                x=list(range(horizon_yrs+1)),
                y=yearly_dists[:,j],
                stackgroup="one", name=state,
                line=dict(width=0.5),
                fillcolor=colors_rating[j]+"cc"
            ))
        fig5.update_layout(xaxis_title="Years", yaxis_title="Probability",
                            yaxis_tickformat=".1%", margin=dict(t=20,b=20),
                            legend=dict(orientation="h"))
        st.plotly_chart(fig5, use_container_width=True)

        # N-year table
        st.subheader("N-Year Default Probabilities")
        check_years = [1,3,5,7,10]
        check_years = [y for y in check_years if y <= horizon_yrs]
        rows_def = []
        for rating in STATES[:-1]:
            r_idx = STATES.index(rating)
            defs  = []
            for yr in check_years:
                P_n = np.linalg.matrix_power(P, yr)
                defs.append(f"{P_n[r_idx, D_IDX]:.3%}")
            rows_def.append([rating]+defs)
        df_def = pd.DataFrame(rows_def, columns=["Rating"]+[f"{y}yr" for y in check_years])
        st.dataframe(df_def.set_index("Rating"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO EXPECTED LOSS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Portfolio Credit Risk — Expected Loss")
    st.markdown(
        "**Expected Loss = PD × LGD × EAD**. "
        "The PD (probability of default) comes from the Markov chain; "
        "LGD = 1 − recovery rate; EAD is the notional exposure."
    )

    st.markdown("**Enter portfolio positions:**")
    port_data = {
        "Bond": ["Bond A","Bond B","Bond C","Bond D","Bond E"],
        "Rating": ["AAA","BBB","BB","BBB","B"],
        "Notional ($M)": [50,100,30,80,40],
    }
    port_df = pd.DataFrame(port_data)
    edited  = st.data_editor(port_df, num_rows="dynamic", use_container_width=True)

    horizon_port = st.slider("Horizon (years)", 1, 10, 3, key="port_horizon")
    LGD = 1 - recovery

    if not edited.empty:
        rows_out = []
        total_el = 0
        for _, row in edited.iterrows():
            try:
                r_idx = STATES.index(row["Rating"])
                P_n   = np.linalg.matrix_power(P, horizon_port)
                pd_   = P_n[r_idx, D_IDX]
                ead   = float(row["Notional ($M)"])
                el    = pd_ * LGD * ead
                total_el += el
                rows_out.append({
                    "Bond":row["Bond"], "Rating":row["Rating"],
                    "EAD ($M)":f"${ead:.1f}",
                    f"PD ({horizon_port}yr)":f"{pd_:.3%}",
                    "LGD":f"{LGD:.0%}",
                    "Expected Loss ($M)":f"${el:.3f}"
                })
            except:
                pass

        if rows_out:
            df_out = pd.DataFrame(rows_out)
            st.dataframe(df_out.set_index("Bond"), use_container_width=True)
            total_ead = edited["Notional ($M)"].sum()
            c1,c2,c3 = st.columns(3)
            c1.metric("Total EAD", f"${total_ead:.1f}M")
            c2.metric(f"Total Expected Loss ({horizon_port}yr)", f"${total_el:.3f}M")
            c3.metric("EL as % of Portfolio", f"{total_el/total_ead:.3%}")

            # EL bar chart
            bonds = [r["Bond"] for r in rows_out]
            els   = [float(r["Expected Loss ($M)"].replace("$","")) for r in rows_out]
            fig6  = px.bar(x=bonds, y=els, title="Expected Loss by Bond",
                            labels={"x":"Bond","y":"Expected Loss ($M)"},
                            color=els, color_continuous_scale="Reds")
            fig6.update_layout(margin=dict(t=40,b=20))
            st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STRESS TESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Stress Testing — Downturn Migration")
    st.markdown(
        "Under economic stress, rating migration accelerates. This simulates "
        "a **stressed transition matrix** by scaling the off-diagonal (migration) "
        "probabilities upward, then renormalizing. "
        "This is analogous to what Basel II/III stress tests do to PD estimates."
    )

    stress_scalar = st.slider(
        "Migration Stress Scalar", 1.0, 5.0, 2.0, step=0.1,
        help="How much to amplify off-diagonal transitions. 1.0 = baseline; 3.0 = severe recession."
    )
    stress_horizon = st.slider("Horizon (years)", 1, 10, 5, key="stress_horizon")
    stress_rating  = st.selectbox("Starting Rating", STATES[:-1], index=3, key="stress_rate")

    # Build stressed matrix
    P_stress = P.copy()
    for i in range(n_states):
        if i == D_IDX: continue
        diag_val    = P_stress[i,i]
        off_diag    = P_stress[i,:].copy(); off_diag[i] = 0
        off_stressed= off_diag * stress_scalar
        overflow    = off_stressed.sum()
        if overflow >= 1:
            off_stressed = off_stressed / (overflow + 1e-8) * 0.999
            overflow = off_stressed.sum()
        P_stress[i,i]  = max(0, 1 - overflow)
        P_stress[i,[j for j in range(n_states) if j!=i]] = off_stressed[[j for j in range(n_states) if j!=i]]
    P_stress[D_IDX] = 0; P_stress[D_IDX, D_IDX] = 1.0

    sr_idx = STATES.index(stress_rating)

    base_cum    = [np.linalg.matrix_power(P,       yr)[sr_idx, D_IDX] for yr in range(stress_horizon+1)]
    stressed_cum= [np.linalg.matrix_power(P_stress, yr)[sr_idx, D_IDX] for yr in range(stress_horizon+1)]

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=list(range(stress_horizon+1)), y=base_cum,
                               name="Baseline", mode="lines+markers",
                               line=dict(color="#4C78A8",width=2)))
    fig7.add_trace(go.Scatter(x=list(range(stress_horizon+1)), y=stressed_cum,
                               name=f"Stressed (×{stress_scalar:.1f})", mode="lines+markers",
                               line=dict(color="#e74c3c",width=2,dash="dash")))
    fig7.update_layout(
        title=f"Cumulative Default Probability — {stress_rating} — Baseline vs. Stressed",
        xaxis_title="Years", yaxis_title="P(Default)", yaxis_tickformat=".2%",
        margin=dict(t=40,b=20)
    )
    st.plotly_chart(fig7, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        st.subheader("Baseline Matrix")
        fig8 = px.imshow(P, x=STATES, y=STATES, color_continuous_scale="Blues",
                          text_auto=".3f", labels=dict(x="To",y="From"))
        fig8.update_layout(margin=dict(t=20,b=20))
        st.plotly_chart(fig8, use_container_width=True)

    with col_f:
        st.subheader(f"Stressed Matrix (×{stress_scalar:.1f})")
        fig9 = px.imshow(P_stress, x=STATES, y=STATES, color_continuous_scale="Reds",
                          text_auto=".3f", labels=dict(x="To",y="From"))
        fig9.update_layout(margin=dict(t=20,b=20))
        st.plotly_chart(fig9, use_container_width=True)

    st.subheader("Default Probability Delta — Stressed vs. Baseline")
    delta_rows = []
    for rating in STATES[:-1]:
        r_idx = STATES.index(rating)
        P_n_b = np.linalg.matrix_power(P,       stress_horizon)
        P_n_s = np.linalg.matrix_power(P_stress, stress_horizon)
        delta_rows.append({
            "Rating":rating,
            "Baseline PD":f"{P_n_b[r_idx,D_IDX]:.3%}",
            "Stressed PD":f"{P_n_s[r_idx,D_IDX]:.3%}",
            "Delta":f"+{P_n_s[r_idx,D_IDX]-P_n_b[r_idx,D_IDX]:.3%}",
            "Stress Multiplier":f"{P_n_s[r_idx,D_IDX]/max(P_n_b[r_idx,D_IDX],1e-8):.1f}×"
        })
    st.dataframe(pd.DataFrame(delta_rows).set_index("Rating"), use_container_width=True)

st.caption(
    "Built with Python · NumPy · Pandas · SciPy · Streamlit · Plotly · "
    "Transition matrix: S&P Global Annual Default Study 2023"
)
