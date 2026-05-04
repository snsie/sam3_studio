import { useState } from "react";

import { AppHeader, type AppPage } from "./components/AppHeader";
import type {
  ActiveChallengeHeaderState,
  LeaderboardHeaderState,
  SandboxHeaderState,
} from "./headerContext";
import { ChallengesPage } from "./pages/ChallengesPage";
import { LeaderboardPage } from "./pages/LeaderboardPage";
import { SandboxPage } from "./pages/SandboxPage";

export default function App() {
  const [activePage, setActivePage] = useState<AppPage>("sandbox");
  const [sandboxState, setSandboxState] = useState<SandboxHeaderState | null>(null);
  const [activeChallengeState, setActiveChallengeState] = useState<ActiveChallengeHeaderState | null>(null);
  const [leaderboardState, setLeaderboardState] = useState<LeaderboardHeaderState | null>(null);

  return (
    <div className={`app-shell${activePage === "sandbox" ? " is-sandbox" : ""}`}>
      <div className="background-grid" aria-hidden="true" />
      <AppHeader
        activePage={activePage}
        onPageChange={setActivePage}
        sandboxState={sandboxState}
        activeChallengeState={activeChallengeState}
        leaderboardState={leaderboardState}
      />

      <section className={`page-view${activePage === "sandbox" ? " is-active" : ""}`} hidden={activePage !== "sandbox"}>
        <SandboxPage onHeaderStateChange={setSandboxState} />
      </section>
      <section className={`page-view${activePage === "challenges" ? " is-active" : ""}`} hidden={activePage !== "challenges"}>
        <ChallengesPage onHeaderStateChange={setActiveChallengeState} />
      </section>
      <section className={`page-view${activePage === "leaderboard" ? " is-active" : ""}`} hidden={activePage !== "leaderboard"}>
        <LeaderboardPage
          isActive={activePage === "leaderboard"}
          onHeaderStateChange={setLeaderboardState}
        />
      </section>
    </div>
  );
}
