import { useState, useEffect } from 'react';

interface Figure {
    filename: string;
    title: string;
    description: string;
    available: boolean;
}

interface DatasetInfo {
    rows: number;
    columns: number;
    column_names: string[];
}

interface ReportsData {
    figures: Figure[];
    model_trained: boolean;
    dataset_loaded: boolean;
    dataset_info: DatasetInfo | null;
}

interface DatasetPreview {
    columns: string[];
    preview: Record<string, unknown>[];
    stats: {
        total_rows: number;
        total_columns: number;
        target_distribution: Record<string, number> | null;
        missing_values: Record<string, number>;
    };
}

interface ClassMetrics {
    precision: number;
    recall: number;
    f1_score: number;
    support: number;
}

interface ModelMetrics {
    metrics: {
        accuracy: number;
        precision: number;
        recall: number;
        f1_score: number;
        roc_auc: number;
        test_samples: number;
        train_samples: number;
    };
    classification_report: {
        normal: ClassMetrics;
        heart_disease: ClassMetrics;
        macro_avg: ClassMetrics;
        weighted_avg: ClassMetrics;
    };
}

const API_BASE = 'http://localhost:8000';

function ReportsPage() {
    const [reports, setReports] = useState<ReportsData | null>(null);
    const [preview, setPreview] = useState<DatasetPreview | null>(null);
    const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedImage, setSelectedImage] = useState<string | null>(null);

    useEffect(() => {
        fetchReports();
    }, []);

    const fetchReports = async () => {
        try {
            setLoading(true);
            const res = await fetch(`${API_BASE}/reports`);
            if (!res.ok) throw new Error('Failed to fetch reports');
            const data = await res.json();
            setReports(data);

            // Fetch dataset preview
            try {
                const previewRes = await fetch(`${API_BASE}/reports/dataset-preview`);
                if (previewRes.ok) {
                    const previewData = await previewRes.json();
                    setPreview(previewData);
                }
            } catch {
                // Dataset preview optional
            }

            // Fetch model metrics
            try {
                const metricsRes = await fetch(`${API_BASE}/reports/model-metrics`);
                if (metricsRes.ok) {
                    const metricsData = await metricsRes.json();
                    setModelMetrics(metricsData);
                }
            } catch {
                // Metrics optional
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load reports');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="reports-page">
                <h2>Model Reports & Analysis</h2>
                <div className="loading">Loading reports...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="reports-page">
                <h2>Model Reports & Analysis</h2>
                <div className="error-message">
                    <strong>Error:</strong> {error}
                    <p>Make sure the backend is running at {API_BASE}</p>
                </div>
            </div>
        );
    }

    const availableFigures = reports?.figures.filter(f => f.available) || [];
    const unavailableFigures = reports?.figures.filter(f => !f.available) || [];

    return (
        <div className="reports-page">
            <h2>Model Reports & Analysis</h2>
            <p className="page-description">
                View generated visualizations, model performance metrics, and dataset statistics.
            </p>

            {/* Status Cards */}
            <div className="status-cards">
                <div className={`status-card ${reports?.model_trained ? 'status-good' : 'status-warning'}`}>
                    <div className="status-icon">{reports?.model_trained ? '✓' : '!'}</div>
                    <div className="status-text">
                        <strong>Model</strong>
                        <span>{reports?.model_trained ? 'Trained' : 'Not trained yet'}</span>
                    </div>
                </div>
                <div className={`status-card ${reports?.dataset_loaded ? 'status-good' : 'status-warning'}`}>
                    <div className="status-icon">{reports?.dataset_loaded ? '✓' : '!'}</div>
                    <div className="status-text">
                        <strong>Dataset</strong>
                        <span>{reports?.dataset_info ? `${reports.dataset_info.rows} rows` : 'Not loaded'}</span>
                    </div>
                </div>
                <div className={`status-card ${availableFigures.length > 0 ? 'status-good' : 'status-warning'}`}>
                    <div className="status-icon">{availableFigures.length}</div>
                    <div className="status-text">
                        <strong>Figures</strong>
                        <span>Available</span>
                    </div>
                </div>
            </div>

            {/* Model Performance Metrics */}
            {modelMetrics && (
                <section className="report-section">
                    <h3>Model Performance</h3>

                    {/* Overall Metrics */}
                    <div className="dataset-stats">
                        <div className="stat-item">
                            <span className="stat-value">{(modelMetrics.metrics.accuracy * 100).toFixed(1)}%</span>
                            <span className="stat-label">Accuracy</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-value">{(modelMetrics.metrics.roc_auc * 100).toFixed(1)}%</span>
                            <span className="stat-label">ROC-AUC</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-value">{modelMetrics.metrics.train_samples}</span>
                            <span className="stat-label">Train Samples</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-value">{modelMetrics.metrics.test_samples}</span>
                            <span className="stat-label">Test Samples</span>
                        </div>
                    </div>

                    {/* Classification Report Table */}
                    <h4>Classification Report</h4>
                    <div className="table-container">
                        <table className="metrics-table">
                            <thead>
                                <tr>
                                    <th>Class</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1-Score</th>
                                    <th>Support</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Normal (0)</strong></td>
                                    <td>{modelMetrics.classification_report.normal.precision.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.normal.recall.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.normal.f1_score.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.normal.support}</td>
                                </tr>
                                <tr>
                                    <td><strong>Heart Disease (1)</strong></td>
                                    <td>{modelMetrics.classification_report.heart_disease.precision.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.heart_disease.recall.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.heart_disease.f1_score.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.heart_disease.support}</td>
                                </tr>
                                <tr className="table-divider">
                                    <td><em>Macro Avg</em></td>
                                    <td>{modelMetrics.classification_report.macro_avg.precision.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.macro_avg.recall.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.macro_avg.f1_score.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.macro_avg.support}</td>
                                </tr>
                                <tr>
                                    <td><em>Weighted Avg</em></td>
                                    <td>{modelMetrics.classification_report.weighted_avg.precision.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.weighted_avg.recall.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.weighted_avg.f1_score.toFixed(4)}</td>
                                    <td>{modelMetrics.classification_report.weighted_avg.support}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <p className="metrics-note">
                        <strong>Precision:</strong> Of predicted positives, how many are correct.
                        <strong> Recall:</strong> Of actual positives, how many were found.
                        <strong> F1-Score:</strong> Harmonic mean of precision and recall.
                    </p>
                </section>
            )}

            {/* Dataset Preview */}
            {preview && (
                <section className="report-section">
                    <h3>Dataset Overview</h3>
                    <div className="dataset-stats">
                        <div className="stat-item">
                            <span className="stat-value">{preview.stats.total_rows}</span>
                            <span className="stat-label">Total Rows</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-value">{preview.stats.total_columns}</span>
                            <span className="stat-label">Features</span>
                        </div>
                        {preview.stats.target_distribution && (
                            <>
                                <div className="stat-item">
                                    <span className="stat-value">{preview.stats.target_distribution[0] || 0}</span>
                                    <span className="stat-label">Normal (0)</span>
                                </div>
                                <div className="stat-item">
                                    <span className="stat-value">{preview.stats.target_distribution[1] || 0}</span>
                                    <span className="stat-label">Heart Disease (1)</span>
                                </div>
                            </>
                        )}
                    </div>

                    <h4>Data Preview (First 10 Rows)</h4>
                    <div className="table-container">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    {preview.columns.map((col) => (
                                        <th key={col}>{col}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {preview.preview.map((row, idx) => (
                                    <tr key={idx}>
                                        {preview.columns.map((col) => (
                                            <td key={col}>{String(row[col] ?? '')}</td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>
            )}

            {/* Available Figures */}
            {availableFigures.length > 0 && (
                <section className="report-section">
                    <h3>Generated Visualizations</h3>
                    <div className="figures-grid">
                        {availableFigures.map((fig) => (
                            <div
                                key={fig.filename}
                                className="figure-card"
                                onClick={() => setSelectedImage(fig.filename)}
                            >
                                <img
                                    src={`${API_BASE}/reports/figures/${fig.filename}`}
                                    alt={fig.title}
                                    loading="lazy"
                                />
                                <div className="figure-info">
                                    <h4>{fig.title}</h4>
                                    <p>{fig.description}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            )}

            {/* Missing Figures */}
            {unavailableFigures.length > 0 && (
                <section className="report-section">
                    <h3>Pending Visualizations</h3>
                    <p className="muted">Run the training and evaluation scripts to generate these figures:</p>
                    <code className="code-block">
                        python -m src.train --data data/raw/heart_disease.csv<br />
                        python -m src.evaluate --data data/raw/heart_disease.csv --model models/best_model.joblib
                    </code>
                    <ul className="pending-list">
                        {unavailableFigures.map((fig) => (
                            <li key={fig.filename}>
                                <strong>{fig.title}</strong> — {fig.description}
                            </li>
                        ))}
                    </ul>
                </section>
            )}

            {/* Image Modal */}
            {selectedImage && (
                <div className="image-modal" onClick={() => setSelectedImage(null)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <button className="modal-close" onClick={() => setSelectedImage(null)}>×</button>
                        <img
                            src={`${API_BASE}/reports/figures/${selectedImage}`}
                            alt="Full size figure"
                        />
                    </div>
                </div>
            )}
        </div>
    );
}

export default ReportsPage;
