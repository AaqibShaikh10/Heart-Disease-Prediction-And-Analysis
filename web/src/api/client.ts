/**
 * API client for the Heart Disease Prediction backend.
 */

const API_BASE_URL = 'http://localhost:8000';

export interface PredictionRequest {
    age: number;
    sex: number;
    'chest pain type': number;
    'resting bp s': number;
    cholesterol: number;
    'fasting blood sugar': number;
    'resting ecg': number;
    'max heart rate': number;
    'exercise angina': number;
    oldpeak: number;
    'ST slope': number;
}

export interface PredictionResponse {
    prediction: number;
    probability: number;
    threshold: number;
}

export interface HealthResponse {
    status: string;
}

/**
 * Check if the API server is healthy.
 */
export async function checkHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
}

/**
 * Make a heart disease prediction.
 */
export async function predict(data: PredictionRequest): Promise<PredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Prediction failed: ${response.status}`);
    }

    return response.json();
}
