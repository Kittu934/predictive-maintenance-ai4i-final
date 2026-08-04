# Chapter 3 Implementation

This chapter explains how the comparative predictive-maintenance system was implemented using the AI4I 2020 dataset. The implementation comprised data validation and preprocessing, development of Random Forest, Support Vector Machine (SVM) and Long Short-Term Memory (LSTM) models, predictive and computational evaluation, explainability analysis, and integration of the outputs into a Streamlit dashboard. The work followed the quantitative comparative design introduced in Chapter 1 and the technical decisions justified in Chapter 2.

## 3.1 System Design

The system was developed as a modular pipeline so that data preparation, model training, evaluation, explanation and presentation could be executed and tested independently. Figure 1 presents the overall architecture.

**[Insert Figure 1: `outputs/figures/figure_1_system_architecture.png`]**

The AI4I 2020 CSV file first passed through validation and preprocessing. The processed observations were divided into training and test sets before being supplied to the three models. Each model generated class scores and predictions, which were used to calculate predictive metrics, computational-efficiency measures and SHAP explanations. The saved models and outputs were then connected to the Streamlit dashboard.

**Table 3. Main implementation components**

| Component | Purpose | Technology |
|---|---|---|
| Data processing | Validate, encode, scale and split the dataset | Python, pandas and scikit-learn |
| Conventional modelling | Train and tune Random Forest and SVM | Scikit-learn [19] |
| Deep-learning modelling | Implement the exploratory one-step LSTM | PyTorch [23] |
| Evaluation | Calculate predictive and efficiency measures | Scikit-learn, Python and psutil |
| Explainability | Generate feature-importance and SHAP outputs | SHAP [10] |
| Presentation | Provide interactive data and model comparison | Streamlit [21] |

## 3.2 Development Environment and Reproducibility

The experiments were executed using Python 3.13.5 on a Linux environment with an AMD EPYC 9V74 processor and approximately 5.93 GB of available RAM. The principal package versions were pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, PyTorch 2.10.0, SHAP 0.50.0, matplotlib 3.10.8, psutil 7.2.2 and joblib 1.5.3. The dashboard dependency was pinned to Streamlit 1.49.1 for reproducible deployment. These details are recorded in the project metadata file.

Scikit-learn provided preprocessing pipelines, cross-validation, Random Forest, SVM and predictive metrics [19]. PyTorch was used for the LSTM because it supported a transparent CPU-based recurrent implementation within the available execution environment [23]. SHAP generated global and local feature-attribution outputs [10], while Streamlit provided the user-facing dashboard [21]. A fixed random seed of 42 was applied where supported. The package requirements, source code, trained models and generated outputs were stored together to support reproducibility.

## 3.3 Dataset Validation and Inspection

The AI4I 2020 dataset was obtained as `ai4i2020.csv` from the UCI Machine Learning Repository [17]. Inspection confirmed 10,000 rows and 14 columns, with no missing values and no duplicated rows. The target contained 9,661 normal observations and 339 machine-failure observations, meaning that failures represented only 3.39% of the dataset. Figure 2 shows this imbalance.

**[Insert Figure 2: `outputs/figures/figure_2_class_distribution.png`]**

The dataset included the numerical condition variables air temperature, process temperature, rotational speed, torque and tool wear, together with the categorical Product Type variable [17], [18]. UDI and Product ID were removed because they functioned as identifiers rather than condition measurements. Product Type was retained and one-hot encoded.

The failure-mode variables TWF, HDF, PWF, OSF and RNF were excluded from the predictors. These variables describe specific failure events from which the overall Machine failure target is derived [17], [18]. Including them would expose target-related information and create data leakage. The final predictor set therefore contained Product Type and five operational variables.

**Table 4. Treatment of AI4I variables**

| Variable | Decision | Reason |
|---|---|---|
| UDI and Product ID | Removed | Identifier fields without condition-monitoring meaning |
| Type | One-hot encoded | Retains product-quality category |
| Five operational variables | Retained | Represent machine condition and operation |
| TWF, HDF, PWF, OSF and RNF | Removed | Prevent target leakage |
| Machine failure | Retained as target | Binary outcome for classification |

## 3.4 Data Preparation and Splitting

The dataset was divided using an 80:20 stratified split with random state 42. The training set contained 8,000 observations, including 271 failures, while the untouched test set contained 2,000 observations, including 68 failures. Stratification preserved the original class proportions.

Preprocessing was fitted only on training data to prevent information from the test set influencing model development [13]. Product Type was one-hot encoded for all models. Random Forest received unscaled numerical inputs because tree-based decisions are not based on numerical distance [7]. SVM and LSTM received standardised numerical variables because both optimisation procedures are sensitive to differences in feature scale [8], [13].

Class imbalance was addressed through model weighting rather than altering the test distribution. Random Forest and SVM used balanced class weights. The LSTM used a positive-class weight within binary cross-entropy loss. This increased the penalty for missed failure cases while preserving the original observations.

## 3.5 Random Forest Implementation

Random Forest was implemented as a scikit-learn pipeline combining one-hot encoding with the classifier [7], [19]. Hyperparameters were selected through three-fold stratified cross-validation using F1-score as the optimisation criterion. The search evaluated 200 and 400 trees, unrestricted and restricted tree depths, and minimum leaf sizes of one and two.

The selected model used 200 trees, unrestricted maximum depth, a minimum leaf size of two, square-root feature sampling and balanced class weights. The configuration provided a stronger validation F1-score than the alternative tested combinations. The model was then refitted using the complete training set.

## 3.6 Support Vector Machine Implementation

SVM was implemented as a pipeline combining one-hot encoding, numerical standardisation and an RBF-kernel classifier [8], [19]. Three-fold stratified cross-validation evaluated C values of 0.1, 1, 10 and 100 together with gamma values of scale, 0.01, 0.1 and 1. The selected configuration used an RBF kernel, C = 100, gamma = scale and balanced class weights. Probability estimation was enabled to support ROC and AUC analysis.

The comparatively high C value reduced regularisation and allowed a more flexible decision boundary. However, this also increased the importance of evaluating generalisation on the untouched test set rather than relying on training performance [8].

## 3.7 LSTM Implementation

The LSTM was implemented in PyTorch with 16 hidden units, a dropout rate of 0.2, the Adam optimiser, a learning rate of 0.001, a batch size of 64 and 60 training epochs [9], [23]. The output layer contained one unit, and binary predictions were generated from sigmoid-transformed logits.

AI4I does not provide genuine machine-specific longitudinal histories [17], [18]. Adjacent rows therefore could not be treated as a real degradation sequence. To avoid creating a misleading artificial time series, each observation was reshaped as one time step with eight encoded and standardised features. The LSTM was consequently implemented as an exploratory deep-learning comparator rather than as a full temporal degradation model. Its findings are interpreted only within this benchmark context.

## 3.8 Decision Threshold Selection

The default threshold of 0.5 is not necessarily optimal for an imbalanced failure-detection task. A validation subset was therefore created from the training data, and the decision threshold for each model was selected by maximising validation F1-score. No test-set outcomes were used for threshold selection.

The resulting thresholds were 0.434 for Random Forest, 0.343 for SVM and 0.912 for LSTM. The high LSTM threshold reflected the probability distribution produced after positive-class weighting. These fixed thresholds were applied once to the untouched test-set scores.

## 3.9 Predictive Evaluation

The three models were evaluated on the same 2,000 test observations. Accuracy, precision, recall, F1-score and AUC-ROC were calculated because accuracy alone can conceal poor detection of the minority failure class [12]. Confusion matrices and ROC curves were also generated. The evaluation code saved a complete prediction file containing the true outcome, model probability and final class prediction for every test observation.

Accuracy measured overall correctness, precision measured the proportion of predicted failures that were genuine failures, and recall measured the proportion of actual failures detected [12]. F1-score balanced precision and recall, while AUC-ROC assessed ranking performance across thresholds [12]. The genuine results are presented and interpreted in Chapter 4.

## 3.10 Computational-Efficiency Measurement

Computational efficiency was assessed using final-model training time, inference time across the 2,000-observation test set and peak process-memory increase. Hyperparameter-search time was excluded from final training time because search spaces differed between models. Timing was repeated and the median was retained to reduce temporary system variation.

Peak memory was measured in fresh model-training processes so that the increase above the post-import baseline could be recorded consistently. The hardware and package versions accompany the values because training time and memory use are environment-dependent. The results therefore support comparison within this experiment rather than universal claims about hardware performance.

## 3.11 Explainability Implementation

Random Forest built-in impurity importance was generated as an initial global indicator [7]. Rotational speed, torque and tool wear received the highest built-in importance values. Because impurity importance does not provide a complete explanation of individual predictions, SHAP was also applied [10].

TreeExplainer was used for Random Forest, KernelExplainer was used on a controlled SVM sample, and GradientExplainer was applied to the PyTorch LSTM. Global summary plots were generated for all three models. A local Random Forest waterfall plot was also produced for an actual failure observation. Sampling was restricted for SVM and LSTM explanations because model-agnostic and gradient explanation can add substantial computational cost.

SHAP values were interpreted as descriptions of fitted model behaviour, not as evidence that a feature causes failure [10]. This distinction was also displayed within the dashboard.

## 3.12 Streamlit Dashboard

The final artefact was implemented as a multi-tab Streamlit dashboard [21]. The interface provides:

- a project overview and architecture diagram;
- default AI4I data selection or compatible CSV upload;
- dataset preview, dimensions, missing-value inspection and class distribution;
- predictions from Random Forest, SVM and LSTM;
- downloadable prediction results;
- comparative performance metrics, ROC curves and confusion matrices;
- Random Forest feature importance and SHAP plots;
- training time, inference time and peak-memory comparisons;
- explicit limitations concerning synthetic data, temporal structure and causal interpretation.

Uploaded files are validated before prediction. The interface checks for the six required feature columns, acceptable Type values and valid numerical data. Invalid inputs produce a clear error rather than an unhandled failure. Trained models and data are cached to prevent unnecessary reloading when the dashboard refreshes.

## 3.13 Testing and Quality Assurance

Functional tests checked data loading, feature validation, model loading, prediction length, probability ranges, threshold application and file download. The pipeline confirmed that failure-mode columns were excluded and that preprocessing objects were fitted using training data only. Saved models were reloaded and used to reproduce predictions.

The dashboard acceptance criteria were based directly on the supervisor’s requested functions.

**Table 5. Dashboard acceptance criteria**

| Requirement | Acceptance criterion | Outcome |
|---|---|---|
| Upload or select data | Default data load and compatible CSV upload work | Met |
| Compare models | Outputs from all three models are available | Met |
| Visualise evaluation metrics | Metrics, ROC curves and confusion matrices display | Met |
| Feature importance and SHAP | Global and local explanation outputs display | Met |
| Efficiency presentation | Training, inference and memory results display | Met |
| Clear error handling | Invalid input produces an understandable message | Met |

A formal participant-based usability study was not conducted because the project was designed without human participants. The interface was therefore evaluated through functional acceptance testing. This limits conclusions about user experience and is addressed in Chapter 4.

## 3.14 Project Management

Implementation was organised into milestones covering literature consolidation, environment preparation, data processing, conventional modelling, LSTM development, evaluation, explainability, dashboard construction and documentation. Evidence of the Gantt chart and Kanban workflow should be included in Appendix B. The code, dataset-access instructions, trained models and artefact files should be linked in Appendix C.

## 3.15 Chapter Summary

This chapter described the completed implementation of the comparative predictive-maintenance system. A leakage-controlled and stratified pipeline was used to develop Random Forest, SVM and an exploratory one-step LSTM. The models were evaluated using predictive performance, computational efficiency and explainability. Their outputs were integrated into a Streamlit dashboard that supports data upload, model comparison, visualisation and explanation. Chapter 4 presents the genuine experimental results and critically evaluates the strengths and limitations of the artefact.
