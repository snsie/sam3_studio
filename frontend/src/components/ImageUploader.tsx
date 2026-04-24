import { useRef, useState } from "react";

interface ImageUploaderProps {
  disabled: boolean;
  currentFilename: string | null;
  onUpload: (file: File) => Promise<void>;
}

export function ImageUploader({
  disabled,
  currentFilename,
  onUpload,
}: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file || disabled) {
      return;
    }
    await onUpload(file);
  };

  return (
    <section className="panel-block">
      <div className="section-heading">
        <span className="section-kicker">Input</span>
        <h2>Image Upload</h2>
      </div>
      <button
        type="button"
        className={`upload-dropzone ${isDragging ? "is-dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={async (event) => {
          event.preventDefault();
          setIsDragging(false);
          await handleFiles(event.dataTransfer.files);
        }}
        disabled={disabled}
      >
        <span className="upload-title">Drop a local image</span>
        <span className="upload-subtitle">
          PNG, JPG, JPEG, or WEBP. Click to browse.
        </span>
        {currentFilename ? (
          <span className="upload-current">Current: {currentFilename}</span>
        ) : null}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        hidden
        onChange={async (event) => {
          await handleFiles(event.target.files);
          event.currentTarget.value = "";
        }}
      />
    </section>
  );
}
