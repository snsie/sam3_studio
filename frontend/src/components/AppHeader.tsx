import { ImageUploader } from "./ImageUploader";
import {
  HEADER_LEADERBOARD_OPTIONS,
  type ActiveChallengeHeaderState,
  type LeaderboardHeaderState,
  type SandboxHeaderState,
} from "../headerContext";

type AppPage = "sandbox" | "challenges" | "leaderboard";

interface AppHeaderProps {
  activePage: AppPage;
  onPageChange: (page: AppPage) => void;
  sandboxState: SandboxHeaderState | null;
  activeChallengeState: ActiveChallengeHeaderState | null;
  leaderboardState: LeaderboardHeaderState | null;
}

const PAGE_OPTIONS: Array<{ id: AppPage; label: string }> = [
  { id: "sandbox", label: "Sandbox" },
  { id: "challenges", label: "Challenges" },
  { id: "leaderboard", label: "Leaderboard" },
];

function getLeaderboardStatusCopy(leaderboardState: LeaderboardHeaderState) {
  if (leaderboardState.errorMessage) {
    return leaderboardState.errorMessage;
  }

  if (leaderboardState.isLoading) {
    return "Loading leaderboard...";
  }

  if (leaderboardState.topEntry) {
    return `Top score: ${leaderboardState.topEntry.user_name} - ${leaderboardState.topEntry.overall_score.toFixed(1)}`;
  }

  return "No submissions have been recorded for this challenge yet.";
}

export function AppHeader({
  activePage,
  onPageChange,
  sandboxState,
  activeChallengeState,
  leaderboardState,
}: AppHeaderProps) {
  const headerPane =
    activePage === "sandbox" && sandboxState ? (
      <section className="app-header-pane app-header-sandbox-pane">
        <ImageUploader
          disabled={sandboxState.disabled}
          currentFilename={sandboxState.currentFilename}
          onUpload={sandboxState.onUpload}
          variant="header"
        />
      </section>
    ) : activePage === "challenges" && activeChallengeState ? (
      <section className="app-header-pane app-header-challenge-pane">
        <div className="app-header-pane-copy">
          {/* <span className="section-kicker">Active Challenge</span> */}
          <h2>{activeChallengeState.title}</h2>
          <p className="app-header-pane-note">{activeChallengeState.description}</p>
        </div>
        <button
          type="button"
          className="secondary-button compact-button challenge-session-button"
          onClick={activeChallengeState.onBackToMenu}
        >
          Back to Challenge Menu
        </button>
      </section>
    ) : activePage === "leaderboard" && leaderboardState ? (
      <section className="app-header-pane app-header-leaderboard-pane">
        <div className="app-header-pane-copy">
          <span className="section-kicker">Leaderboard</span>
          <h2>Compare Results</h2>
          <p className={`app-header-pane-note${leaderboardState.errorMessage ? " is-error" : ""}`}>
            {getLeaderboardStatusCopy(leaderboardState)}
          </p>
        </div>
        <div className="inline-toggle-group app-header-pane-actions">
          {HEADER_LEADERBOARD_OPTIONS.map((option) => {
            const isActive = option.id === leaderboardState.activeChallengeId;
            return (
              <button
                key={option.id}
                type="button"
                className={`inline-toggle-button${isActive ? " is-active" : ""}`}
                onClick={() => leaderboardState.onChallengeChange(option.id)}
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </section>
    ) : null;

  return (
    <header className="hero-header app-header-shell">
      <div className="header-copy">
        <span className="section-kicker">Workshop Studio</span>
        <h1>SAM3 Studio</h1>
      </div>
      <div className="app-header-nav">
        <div className="page-switcher" role="tablist" aria-label="App pages">
          {PAGE_OPTIONS.map((option) => {
            const isActive = option.id === activePage;
            return (
              <button
                key={option.id}
                type="button"
                role="tab"
                aria-selected={isActive}
                className={`page-switcher-button${isActive ? " is-active" : ""}`}
                onClick={() => onPageChange(option.id)}
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </div>
      <div className="app-header-context" aria-live="polite">
        {headerPane}
      </div>
    </header>
  );
}

export type { AppPage };