interface ResultCardProps {
    prediction: number;
    probability: number;
    threshold: number;
}

function ResultCard({ prediction, probability, threshold }: ResultCardProps) {
    const isPositive = prediction === 1;
    const probabilityPercent = (probability * 100).toFixed(1);

    return (
        <div className={`result-card ${isPositive ? 'result-positive' : 'result-negative'}`}>
            <h2 className="result-title">Prediction Result</h2>

            <div className="result-main">
                <div className="result-icon">
                    {isPositive ? '⚠️' : '✓'}
                </div>
                <div className="result-label">
                    {isPositive ? 'Heart Disease Predicted' : 'Normal (No Heart Disease)'}
                </div>
            </div>

            <div className="result-details">
                <div className="result-detail">
                    <span className="detail-label">Probability:</span>
                    <span className="detail-value">{probabilityPercent}%</span>
                </div>
                <div className="result-detail">
                    <span className="detail-label">Threshold:</span>
                    <span className="detail-value">{(threshold * 100).toFixed(0)}%</span>
                </div>
            </div>

            <p className="result-note">
                {isPositive
                    ? 'The model predicts elevated risk. Please consult a healthcare professional.'
                    : 'The model predicts low risk, but regular checkups are still recommended.'}
            </p>
        </div>
    );
}

export default ResultCard;
