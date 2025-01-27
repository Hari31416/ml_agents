# Overview of Machine Learning Agents

The user will provide the features and targets. The agents will to every step of the machine learning process and will return the best model.

## Make it Simple

We will start by making some assumptions about the dataset. The goal is to create a library that is simple, yet modular. Later, we can add more features.

### Assumptions

- Heavy feature engineering is not required. The agent should create some features but user should not depend on it.
- The data is somewhat clean and does not require heavy preprocessing. A completely clean dataset is not necessary, the agent will do some preprocessing.
- The data is in tabular format.
- It is supervised learning. Learning tasks can be:
  - Binary classification
  - Multi-class classification
  - Regression
- No time-series data.
- No textual data.

## Libraries

- **crewAI:** For creating agents. Main dependency.
- **smolagents:** Some tools like local Python executor will be used from this library.
- **Langchain:** Some tools to be used.

## Workflow and Modules

- **Machine Learning Utils:** Some utilities to be used in the process.
- **Tools:** Tools based on the utilities.
- **Tests:** Tests to be run before a task is completed.
- **Agents:** Agents to be used in the process.
- **Tasks:** Tasks to be completed.
- **Crew:** The crew using the agents and tasks.

### `ml_utils`

Utilities for:

- **Preprocessing:**
  - Missing value handling (imputation strategies: mean/median/mode, drop columns/rows)
  - Categorical encoding (one-hot, label, ordinal, target encoding)
  - Feature scaling (standardization, normalization)
  - Outlier detection & handling (IQR, Z-score)
- **Feature Engineering:** Polynomial features, combining two or more features, other transformations on single features like log, exponential, time features.
  - Polynomial feature creation
  - Interaction terms
  - Log and Power transformations
  - Date/time feature decomposition
- **EDA:**
  - Summary statistics
  - correlation
  - multicollinearity.
- **Feature Selection:**
  - Variance thresholding
  - RFE
  - mutual information
  - high outlier/null values.
- **Model Development:**
  - Basic model templates (logistic regression, random forest, XGBoost, etc.)
  - Metric calculators (accuracy, F1, RMSE, R²)
  - Cross-validation strategies
  - Hyperparameter grids

### `ml_tests`

- **Data Tests:**

  - Missing value check
  - Data type validation
  - Class imbalance detection
  - Outlier detection
  - Feature correlation analysis

- **Model Tests:**

  - Baseline accuracy check
  - Cross-validation consistency
  - Overfitting detection (train-test gap)
  - Feature importance sanity check
  - Prediction distribution analysis

- **Implementation:**
  - Returns (success: bool, message: str)
  - Threshold configurations for warnings/errors

### `ml_tools`

Most of the important methods in the `ml_utils` module will be converted into tools.

### Agents

- **DataPreprocessingAgent**

  - Role: "Data Cleaning Specialist"
  - Goal: "Prepare raw data for machine learning"

- **FeatureEngineerAgent**

  - Role: "Feature Optimization Expert"
  - Goal: "Create informative predictive features"

- **ModelArchitectAgent**

  - Role: "Algorithm Selection Specialist"
  - Goal: "Choose optimal base model architecture"

- **HyperparameterOptimizerAgent**

  - Role: "Model Tuning Expert"
  - Goal: "Optimize model hyperparameters"

- **ValidationAgent**

  - Role: "Model Quality Assurance"
  - Goal: "Ensure model robustness"

- **LeadScientistAgent**
  - Role: "Machine Learning Project Manager"
  - Goal: "Orchestrate end-to-end ML pipeline"

### Tasks

- **DataPreprocessingTask**

  - Description: Clean and normalize raw data
  - Agent: DataPreprocessingAgent
  - Expected Output: Processed DataFrame

- **FeatureEngineeringTask**

  - Description: Create enhanced feature set
  - Agent: FeatureEngineerAgent
  - Expected Output: Engineered features

- **ModelSelectionTask**

  - Description: Choose initial model candidates
  - Agent: ModelArchitectAgent
  - Expected Output: Shortlisted models

- **HyperparameterTuningTask**

  - Description: Optimize model parameters
  - Agent: HyperparameterOptimizerAgent
  - Expected Output: Tuned models with scores

- **ModelValidationTask**

  - Description: Final model evaluation
  - Agent: ValidationAgent
  - Expected Output: Validation report

- **DeploymentPreparationTask**
  - Description: Package final model
  - Agent: LeadScientistAgent
  - Expected Output: Serialized model + metadata

## To-Dos

- [x] Integrate tools from smolagents. crewAI is already compatible with langchain.
- [x] Create overview of various modules.
  - [x] Machine Learning Utils
  - [x] Tools
  - [x] Tests
  - [x] Agents
  - [x] Tasks
  - [x] Crew
    - [ ] One or many?
- [ ] How to connect the `ml_tools` with `ml_utils`?
  - [ ] How to handle data type like dataframe, series etc. in tools?
  - [ ] Docstrings? I prefer numpy style, crewAI uses google style.
  - [ ] How to avoid repetition of code?
- [ ] Create utilities for machine learning.
  - [ ] Preprocessing
  - [ ] Feature Engineering
  - [ ] EDA
  - [ ] Feature Selection
  - [ ] Model Development
- [ ] Create tools using the utilities.
- [ ] Create tests for the tasks.
  - [ ] Data Tests
  - [ ] Model Tests
- [ ] Create agents, tasks and crews.
- [ ] Learn new things.
  - [ ] Learn about memory management in crewAI.
