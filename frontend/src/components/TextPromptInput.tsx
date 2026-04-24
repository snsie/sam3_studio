interface TextPromptInputProps {
  disabled: boolean;
  value: string;
  onChange: (value: string) => void;
}

export function TextPromptInput({
  disabled,
  value,
  onChange,
}: TextPromptInputProps) {
  return (
    <section className="panel-block">
      <div className="section-heading">
        <span className="section-kicker">Language</span>
        <h2>Text Prompt</h2>
      </div>
      <label className="field-label" htmlFor="text-prompt">
        Experimental SAM3 grounding prompt
      </label>
      <textarea
        id="text-prompt"
        className="text-input"
        rows={3}
        disabled={disabled}
        placeholder="person wearing a helmet"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      {/* <p className="field-note">
        Any non-empty text prompt is applied automatically. If you only enter
        text, SAM3 Studio runs text grounding. If you also add points or a box,
        the text is used as an additional prior over the interactive masks.
      </p> */}
    </section>
  );
}
