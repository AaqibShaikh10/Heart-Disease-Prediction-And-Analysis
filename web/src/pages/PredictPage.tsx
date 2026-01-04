import { useState, FormEvent } from 'react';
import { predict, PredictionRequest, PredictionResponse } from '../api/client';
import FieldNumber from '../components/FieldNumber';
import FieldSelect from '../components/FieldSelect';
import ResultCard from '../components/ResultCard';

// Options for categorical fields
const SEX_OPTIONS = [
    { value: 0, label: '0 - Female' },
    { value: 1, label: '1 - Male' },
];

const CHEST_PAIN_OPTIONS = [
    { value: 1, label: '1 - Typical Angina' },
    { value: 2, label: '2 - Atypical Angina' },
    { value: 3, label: '3 - Non-anginal Pain' },
    { value: 4, label: '4 - Asymptomatic' },
];

const FASTING_BS_OPTIONS = [
    { value: 0, label: '0 - False (≤120 mg/dl)' },
    { value: 1, label: '1 - True (>120 mg/dl)' },
];

const RESTING_ECG_OPTIONS = [
    { value: 0, label: '0 - Normal' },
    { value: 1, label: '1 - ST-T Abnormality' },
    { value: 2, label: '2 - LVH (Estes criteria)' },
];

const EXERCISE_ANGINA_OPTIONS = [
    { value: 0, label: '0 - No' },
    { value: 1, label: '1 - Yes' },
];

const ST_SLOPE_OPTIONS = [
    { value: 0, label: '0 - Unknown/Other' },
    { value: 1, label: '1 - Upsloping' },
    { value: 2, label: '2 - Flat' },
    { value: 3, label: '3 - Downsloping' },
];

function PredictPage() {
    // Form state
    const [age, setAge] = useState<number>(50);
    const [sex, setSex] = useState<number>(1);
    const [chestPainType, setChestPainType] = useState<number>(1);
    const [restingBpS, setRestingBpS] = useState<number>(120);
    const [cholesterol, setCholesterol] = useState<number>(200);
    const [fastingBloodSugar, setFastingBloodSugar] = useState<number>(0);
    const [restingEcg, setRestingEcg] = useState<number>(0);
    const [maxHeartRate, setMaxHeartRate] = useState<number>(150);
    const [exerciseAngina, setExerciseAngina] = useState<number>(0);
    const [oldpeak, setOldpeak] = useState<number>(0);
    const [stSlope, setStSlope] = useState<number>(1);

    // UI state
    const [result, setResult] = useState<PredictionResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setError(null);
        setResult(null);
        setLoading(true);

        // Basic validation
        if (restingBpS <= 0) {
            setError('Resting blood pressure must be greater than 0');
            setLoading(false);
            return;
        }
        if (cholesterol <= 0) {
            setError('Cholesterol must be greater than 0');
            setLoading(false);
            return;
        }
        if (maxHeartRate <= 0) {
            setError('Max heart rate must be greater than 0');
            setLoading(false);
            return;
        }

        const requestData: PredictionRequest = {
            age,
            sex,
            'chest pain type': chestPainType,
            'resting bp s': restingBpS,
            cholesterol,
            'fasting blood sugar': fastingBloodSugar,
            'resting ecg': restingEcg,
            'max heart rate': maxHeartRate,
            'exercise angina': exerciseAngina,
            oldpeak,
            'ST slope': stSlope,
        };

        try {
            const response = await predict(requestData);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Prediction failed. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="predict-page">
            <h2>Heart Disease Prediction</h2>
            <p className="page-description">
                Enter patient data below to predict heart disease risk. All fields are required.
            </p>

            <form className="predict-form" onSubmit={handleSubmit}>
                <div className="form-section">
                    <h3>Demographics</h3>
                    <div className="form-grid">
                        <FieldNumber
                            id="age"
                            label="Age"
                            value={age}
                            onChange={setAge}
                            min={1}
                            max={120}
                            hint="Years"
                        />
                        <FieldSelect
                            id="sex"
                            label="Sex"
                            value={sex}
                            onChange={setSex}
                            options={SEX_OPTIONS}
                        />
                    </div>
                </div>

                <div className="form-section">
                    <h3>Cardiovascular Measurements</h3>
                    <div className="form-grid">
                        <FieldNumber
                            id="restingBpS"
                            label="Resting Blood Pressure"
                            value={restingBpS}
                            onChange={setRestingBpS}
                            min={1}
                            hint="mm Hg (systolic)"
                        />
                        <FieldNumber
                            id="cholesterol"
                            label="Cholesterol"
                            value={cholesterol}
                            onChange={setCholesterol}
                            min={1}
                            hint="mg/dl (serum)"
                        />
                        <FieldNumber
                            id="maxHeartRate"
                            label="Max Heart Rate"
                            value={maxHeartRate}
                            onChange={setMaxHeartRate}
                            min={1}
                            max={250}
                            hint="bpm (achieved)"
                        />
                        <FieldNumber
                            id="oldpeak"
                            label="Oldpeak"
                            value={oldpeak}
                            onChange={setOldpeak}
                            step={0.1}
                            hint="ST depression (can be negative)"
                        />
                    </div>
                </div>

                <div className="form-section">
                    <h3>Clinical Indicators</h3>
                    <div className="form-grid">
                        <FieldSelect
                            id="chestPainType"
                            label="Chest Pain Type"
                            value={chestPainType}
                            onChange={setChestPainType}
                            options={CHEST_PAIN_OPTIONS}
                        />
                        <FieldSelect
                            id="fastingBloodSugar"
                            label="Fasting Blood Sugar"
                            value={fastingBloodSugar}
                            onChange={setFastingBloodSugar}
                            options={FASTING_BS_OPTIONS}
                        />
                        <FieldSelect
                            id="restingEcg"
                            label="Resting ECG"
                            value={restingEcg}
                            onChange={setRestingEcg}
                            options={RESTING_ECG_OPTIONS}
                        />
                        <FieldSelect
                            id="exerciseAngina"
                            label="Exercise-induced Angina"
                            value={exerciseAngina}
                            onChange={setExerciseAngina}
                            options={EXERCISE_ANGINA_OPTIONS}
                        />
                        <FieldSelect
                            id="stSlope"
                            label="ST Slope"
                            value={stSlope}
                            onChange={setStSlope}
                            options={ST_SLOPE_OPTIONS}
                        />
                    </div>
                </div>

                <div className="form-actions">
                    <button type="submit" className="btn btn-primary" disabled={loading}>
                        {loading ? 'Predicting...' : 'Predict'}
                    </button>
                </div>
            </form>

            {error && (
                <div className="error-message">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {result && (
                <ResultCard
                    prediction={result.prediction}
                    probability={result.probability}
                    threshold={result.threshold}
                />
            )}
        </div>
    );
}

export default PredictPage;
