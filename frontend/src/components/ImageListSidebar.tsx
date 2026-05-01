import type { ChallengeImageItem } from "../types";

interface ImageListSidebarProps {
  images: ChallengeImageItem[];
  activeImageId: string | null;
  completedImageIds: Set<string>;
  highlightedImageId: string | null;
  isLoading: boolean;
  errorMessage: string | null;
  onSelectImage: (imageId: string) => void;
}

export function ImageListSidebar({
  images,
  activeImageId,
  completedImageIds,
  highlightedImageId,
  isLoading,
  errorMessage,
  onSelectImage,
}: ImageListSidebarProps) {
  return (
    <aside className="challenge-image-sidebar panel-block">
      <div className="section-heading">
        <span className="section-kicker">Dataset</span>
        <h2>Image List</h2>
      </div>
      {isLoading ? <p className="field-note">Loading challenge images...</p> : null}
      {errorMessage ? <div className="error-banner inline-error">{errorMessage}</div> : null}
      {!isLoading && !errorMessage ? (
        <div className="challenge-image-list">
          {images.map((image) => {
            const isComplete = completedImageIds.has(image.image_id);
            const isActive = image.image_id === activeImageId;
            const isHighlighted = image.image_id === highlightedImageId;
            return (
              <button
                key={image.image_id}
                type="button"
                className={`challenge-image-item${isActive ? " is-active" : ""}${isHighlighted ? " is-recently-stored" : ""}`}
                onClick={() => onSelectImage(image.image_id)}
              >
                <span className={`challenge-status-dot ${isComplete ? "is-complete" : "is-pending"}`}>
                  {isComplete ? "✓" : "○"}
                </span>
                <span className="challenge-image-copy">
                  <strong>{image.name}</strong>
                  <span>{`Image ${image.index}`}</span>
                </span>
              </button>
            );
          })}
        </div>
      ) : null}
    </aside>
  );
}