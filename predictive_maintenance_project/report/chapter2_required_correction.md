# Required consistency correction in Chapter 2

The completed implementation uses **PyTorch** for the LSTM rather than TensorFlow/Keras. This does not change the approved research question, models, dataset, SHAP analysis or Streamlit artefact.

Replace the current Chapter 2 sentence:

> TensorFlow and Keras support recurrent neural-network implementation and are therefore suitable for the LSTM model, although they introduce greater dependency and configuration complexity than scikit-learn [20].

with:

> PyTorch was selected for the LSTM implementation because it provides flexible recurrent-network components, supports CPU-based experimentation and integrates with the Python data-processing and explainability pipeline [23]. Although TensorFlow/Keras could implement the same architecture [20], PyTorch was more compatible with the final execution environment.

In Table 2, replace the deep-learning row with:

| Project requirement | Selected technology | Alternative technologies | Critical comparison and rationale |
|---|---|---|---|
| Deep-learning implementation | PyTorch [23] | TensorFlow/Keras [20] | Both can implement recurrent neural networks. PyTorch was selected because it supported the final CPU-based environment and enabled direct integration with the LSTM and SHAP workflow. TensorFlow/Keras offers a high-level alternative, but using both frameworks would add unnecessary complexity. |

Reference [23] is already included in the current reference list.
