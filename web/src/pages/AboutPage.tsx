function AboutPage() {
    return (
        <div className="about-page">
            <h2>About This Project</h2>

            <section className="about-section">
                <h3>Project Overview</h3>
                <p>
                    This is a <strong>Heart Disease Prediction</strong> application built as a
                    Data Mining project. It uses machine learning techniques to predict the
                    likelihood of heart disease based on clinical and physiological measurements.
                </p>
                <p>
                    The backend uses a trained scikit-learn pipeline that applies preprocessing
                    (imputation, scaling, one-hot encoding) and classification using the best
                    performing algorithm from model comparison.
                </p>
            </section>

            <section className="about-section">
                <h3>🧠 Algorithms Used</h3>
                <p>This project compares <strong>5 classification algorithms</strong>:</p>
                <table className="data-dictionary">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Algorithm</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>1</td>
                            <td><strong>Naive Bayes</strong></td>
                            <td>Probabilistic classifier based on Bayes' theorem with independence assumption</td>
                        </tr>
                        <tr>
                            <td>2</td>
                            <td><strong>Logistic Regression</strong></td>
                            <td>Linear model for binary classification using sigmoid function</td>
                        </tr>
                        <tr>
                            <td>3</td>
                            <td><strong>Decision Tree (CART)</strong></td>
                            <td>Tree-based classifier using Gini impurity or Information Gain</td>
                        </tr>
                        <tr>
                            <td>4</td>
                            <td><strong>K-Nearest Neighbors (KNN)</strong></td>
                            <td>Distance-based classifier that predicts based on nearest neighbors</td>
                        </tr>
                        <tr>
                            <td>5</td>
                            <td><strong>Random Forest</strong></td>
                            <td>Ensemble of decision trees with bagging (Best performer: ~92% ROC-AUC)</td>
                        </tr>
                    </tbody>
                </table>
            </section>

            <section className="about-section">
                <h3>Dataset Information</h3>
                <p>
                    The dataset contains patient records with the following features:
                </p>
                <table className="data-dictionary">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Description</th>
                            <th>Values</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>age</td>
                            <td>Age of the patient</td>
                            <td>Years (numeric)</td>
                        </tr>
                        <tr>
                            <td>sex</td>
                            <td>Biological sex</td>
                            <td>0 = Female, 1 = Male</td>
                        </tr>
                        <tr>
                            <td>chest pain type</td>
                            <td>Type of chest pain experienced</td>
                            <td>1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic</td>
                        </tr>
                        <tr>
                            <td>resting bp s</td>
                            <td>Resting blood pressure (systolic)</td>
                            <td>mm Hg (numeric)</td>
                        </tr>
                        <tr>
                            <td>cholesterol</td>
                            <td>Serum cholesterol</td>
                            <td>mg/dl (numeric)</td>
                        </tr>
                        <tr>
                            <td>fasting blood sugar</td>
                            <td>Fasting blood sugar &gt; 120 mg/dl</td>
                            <td>0 = False, 1 = True</td>
                        </tr>
                        <tr>
                            <td>resting ecg</td>
                            <td>Resting electrocardiogram results</td>
                            <td>0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy</td>
                        </tr>
                        <tr>
                            <td>max heart rate</td>
                            <td>Maximum heart rate achieved</td>
                            <td>bpm (numeric)</td>
                        </tr>
                        <tr>
                            <td>exercise angina</td>
                            <td>Exercise-induced angina</td>
                            <td>0 = No, 1 = Yes</td>
                        </tr>
                        <tr>
                            <td>oldpeak</td>
                            <td>ST depression induced by exercise relative to rest</td>
                            <td>Numeric (can be negative)</td>
                        </tr>
                        <tr>
                            <td>ST slope</td>
                            <td>Slope of the peak exercise ST segment</td>
                            <td>1 = Upsloping, 2 = Flat, 3 = Downsloping</td>
                        </tr>
                        <tr>
                            <td>target</td>
                            <td>Diagnosis of heart disease</td>
                            <td>0 = No disease, 1 = Heart disease</td>
                        </tr>
                    </tbody>
                </table>
            </section>

            <section className="about-section">
                <h3>Data Processing Notes</h3>
                <ul>
                    <li>Zero values for <code>resting bp s</code> and <code>cholesterol</code> are
                        treated as missing and imputed using median.</li>
                    <li>The <code>oldpeak</code> column may contain negative values, which are kept
                        as-is (not clipped).</li>
                    <li>Duplicate rows are removed during preprocessing.</li>
                    <li>Categorical features are one-hot encoded; numeric features are standardized.</li>
                </ul>
            </section>

            <section className="about-section disclaimer-section">
                <h3>⚠️ Disclaimer</h3>
                <p>
                    <strong>This application is for educational purposes only.</strong> It is not
                    intended to provide medical advice, diagnosis, or treatment recommendations.
                    The predictions made by this model should not be used as a substitute for
                    professional medical consultation.
                </p>
                <p>
                    Always consult a qualified healthcare provider for any health-related concerns
                    or decisions.
                </p>
            </section>

            <section className="about-section">
                <h3>Technical Details</h3>
                <ul>
                    <li><strong>Backend:</strong> Python, FastAPI, scikit-learn</li>
                    <li><strong>Frontend:</strong> React, TypeScript, Vite</li>
                    <li><strong>Model:</strong> Ensemble classifier with preprocessing pipeline</li>
                    <li><strong>Validation:</strong> Stratified 5-fold cross-validation</li>
                </ul>
            </section>
        </div>
    );
}

export default AboutPage;
