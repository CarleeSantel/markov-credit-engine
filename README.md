# Markov Credit Engine
## Corporate credit rating migration modeling, multi-period default probability, and portfolio stress testing

Credit ratings don't stay put — they migrate. This tool models that migration as a Markov chain using S&P's historical transition matrix (1981–2023 averages), then uses Chapman-Kolmogorov matrix exponentiation to compute default probabilities at any horizon. The portfolio layer calculates expected loss across a book of rated credits (PD × LGD × EAD), and the stress testing tab lets you scale migration intensity to simulate recession-style deterioration. Default is modeled as an absorbing state, which means once a credit defaults, it stays there — as it should.

---

## Installation (run it yourself)

Requires Python 3.9+.

```bash
git clone https://github.com/CarleeSantel/markov-credit-engine.git
cd markov-credit-engine
pip install streamlit pandas numpy scipy plotly
streamlit run markov_credit.py
```

---

## Installation (contribute)

```bash
git clone https://github.com/CarleeSantel/markov-credit-engine.git
cd markov-credit-engine
pip install streamlit pandas numpy scipy plotly
```

The transition matrix is hardcoded from S&P published averages. Multi-period dynamics use `numpy.linalg.matrix_power(P, n)`. Stress testing scales off-diagonal elements by a scalar and renormalizes rows — the absorbing default state is held fixed throughout.

---

## Contributing

Open an issue before submitting a pull request. If you're proposing changes to the transition matrix source or stress testing methodology, include a reference to the data or paper you're drawing from.

---

## Known issues

- Transition matrix is static (1981–2023 average) — no support for through-the-cycle vs. point-in-time switching yet
- LGD is user-input only; no market-implied LGD model
- Stress scalar is a blunt instrument — a future version should allow rating-band-specific stress factors
