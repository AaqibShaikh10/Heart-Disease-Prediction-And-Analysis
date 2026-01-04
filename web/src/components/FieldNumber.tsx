interface FieldNumberProps {
    id: string;
    label: string;
    value: number | string;
    onChange: (value: number) => void;
    min?: number;
    max?: number;
    step?: number;
    required?: boolean;
    hint?: string;
}

function FieldNumber({
    id,
    label,
    value,
    onChange,
    min,
    max,
    step = 1,
    required = true,
    hint,
}: FieldNumberProps) {
    return (
        <div className="field">
            <label htmlFor={id} className="field-label">
                {label}
                {required && <span className="required">*</span>}
            </label>
            <input
                type="number"
                id={id}
                className="field-input"
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
                min={min}
                max={max}
                step={step}
                required={required}
            />
            {hint && <span className="field-hint">{hint}</span>}
        </div>
    );
}

export default FieldNumber;
