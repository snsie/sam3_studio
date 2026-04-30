import type { KeyboardEvent } from "react";

interface TextPromptInputProps {
  disabled: boolean;
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

export function TextPromptInput({
  disabled,
  value,
  onChange,
  onSubmit,
}: TextPromptInputProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }

    event.preventDefault();
    onSubmit();
  };

  return (
    <section className="panel-block">
      <div className="section-heading">
        <span className="section-kicker">Language</span>
        <h2>Text Prompt</h2>
      </div>
      <label className="field-label" htmlFor="text-prompt">
        Experimental SAM3 text prompt
      </label>
      <textarea
        id="text-prompt"
        className="text-input"
        rows={3}
        disabled={disabled}
        placeholder="person wearing a helmet"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      {/* <p className="field-note">
        Any non-empty text prompt is applied alongside your active points or
        boxes when supported. SAM3 Studio automatically chooses the best
        available inference path for the current prompt set.
      </p> */}
    </section>
  );
}
