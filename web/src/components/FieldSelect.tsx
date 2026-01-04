interface Option {
    value: number;
    label: string;
}

interface FieldSelectProps {
    id: string;
    label: string;
    value: number;
    onChange: (value: number) => void;
    options: Option[];
    required?: boolean;
    hint?: string;
}

function FieldSelect({
    id,
    label,
    value,
    onChange,
    options,
    required = true,
    hint,
}: FieldSelectProps) {
    return (
        <div className="field">
            <label htmlFor={id} className="field-label">
                {label}
                {required && <span className="required">*</span>}
            </label>
            <select
                id={id}
                className="field-select"
                value={value}
                onChange={(e) => onChange(parseInt(e.target.value, 10))}
                required={required}
            >
                {options.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                        {opt.label}
                    </option>
                ))}
            </select>
            {hint && <span className="field-hint">{hint}</span>}
        </div>
    );
}

export default FieldSelect;
