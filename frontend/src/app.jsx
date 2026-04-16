import { useEffect, useRef, useState } from "react";

const API_BASE = "https://gym-pose-estimation-webapp.onrender.com";

function App() {
  const [exercise, setExercise] = useState("curl");
  const [file, setFile] = useState(null);
  const [saveVideo, setSaveVideo] = useState(true);
  const [savePlots, setSavePlots] = useState(true);
  const [saveAngleCsv, setSaveAngleCsv] = useState(false);
  const [saveRepsJson, setSaveRepsJson] = useState(false);

  const [loading, setLoading] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);
  const [error, setError] = useState("");
  const [analyses, setAnalyses] = useState([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);

  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      text: "Ask about this workout. For example: best rep, ROM, tempo, pass/fail, or whether later reps dropped off.",
    },
  ]);

  const latestDetailRequestRef = useRef(0);
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchAnalyses();
  }, []);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatMessages]);

  async function fetchAnalyses() {
    try {
      setError("");
      const res = await fetch(`${API_BASE}/analyses`);
      const data = await res.json();
      setAnalyses(data.analyses || []);
    } catch (err) {
      console.error(err);
      setError("Could not load saved analyses.");
    }
  }

  async function fetchAnalysisDetail(id) {
    const requestId = ++latestDetailRequestRef.current;

    try {
      setError("");
      const res = await fetch(`${API_BASE}/analyses/${id}`);
      const data = await res.json();

      if (requestId !== latestDetailRequestRef.current) {
        return;
      }

      setSelectedAnalysis(data);
      setChatMessages([
        {
          role: "assistant",
          text: "Ask about this workout. For example: best rep, ROM, tempo, pass/fail, or whether later reps dropped off.",
        },
      ]);
      setChatInput("");
    } catch (err) {
      console.error(err);

      if (requestId === latestDetailRequestRef.current) {
        setError("Could not load analysis detail.");
      }
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!file) {
      setError("Please choose a video file first.");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("exercise", exercise);
    formData.append("calibration_path", "calibration_easy.json");
    formData.append("save_video", saveVideo);
    formData.append("save_plots", savePlots);
    formData.append("save_angle_csv", saveAngleCsv);
    formData.append("save_reps_json", saveRepsJson);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Analysis failed.");
      }

      await refreshAnalysesAndSelect(data.analysis_id);

      setFile(null);
      event.target.reset();
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong while analyzing the video.");
    } finally {
      setLoading(false);
    }
  }

  async function handleRunDemo() {
    setDemoLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("exercise", "curl");
    formData.append("calibration_path", "calibration_easy.json");
    formData.append("save_video", "true");
    formData.append("save_plots", "true");
    formData.append("save_angle_csv", String(saveAngleCsv));
    formData.append("save_reps_json", String(saveRepsJson));

    try {
      const res = await fetch(`${API_BASE}/analyze-demo`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Demo analysis failed.");
      }

      await refreshAnalysesAndSelect(data.analysis_id);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong while running the demo.");
    } finally {
      setDemoLoading(false);
    }
  }

  async function refreshAnalysesAndSelect(analysisId) {
    const analysesRes = await fetch(`${API_BASE}/analyses`);
    const analysesData = await analysesRes.json();
    const updatedAnalyses = analysesData.analyses || [];
    setAnalyses(updatedAnalyses);

    const latestAnalysisId =
      analysisId || (updatedAnalyses.length > 0 ? updatedAnalyses[0].id : null);

    if (latestAnalysisId) {
      await fetchAnalysisDetail(latestAnalysisId);
    }
  }

  async function handleSendChat() {
    const trimmed = chatInput.trim();

    if (!trimmed || !selectedAnalysis?.analysis?.id || chatLoading) {
      return;
    }

    const userMessage = {
      role: "user",
      text: trimmed,
    };

    const updatedMessages = [...chatMessages, userMessage];

    setChatMessages(updatedMessages);
    setChatInput("");
    setChatLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          analysis_id: selectedAnalysis.analysis.id,
          messages: updatedMessages,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Chat request failed.");
      }

      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.reply || "No reply returned.",
        },
      ]);
    } catch (err) {
      console.error(err);
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Something went wrong while generating the reply.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  function handleChatKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSendChat();
    }
  }

  function getArtifactUrl(path) {
    if (!path) return null;
    return `${API_BASE}${path}`;
  }

  function formatExerciseName(name) {
    if (!name) return "Unknown";
    return name.charAt(0).toUpperCase() + name.slice(1);
  }

  function renderSummaryCards() {
    if (!selectedAnalysis) return null;

    const analysis = selectedAnalysis.analysis;

    return (
      <div style={styles.summaryGrid}>
        <div style={styles.summaryCard}>
          <p style={styles.cardLabel}>Exercise</p>
          <h3 style={styles.cardValue}>{formatExerciseName(analysis.exercise)}</h3>
        </div>

        <div style={styles.summaryCard}>
          <p style={styles.cardLabel}>Rep Count</p>
          <h3 style={styles.cardValue}>{analysis.rep_count}</h3>
        </div>

        <div style={styles.summaryCard}>
          <p style={styles.cardLabel}>Pass Count</p>
          <h3 style={{ ...styles.cardValue, color: "#86efac" }}>{analysis.pass_count}</h3>
        </div>

        <div style={styles.summaryCard}>
          <p style={styles.cardLabel}>Fail Count</p>
          <h3 style={{ ...styles.cardValue, color: "#fca5a5" }}>{analysis.fail_count}</h3>
        </div>

        <div style={styles.summaryCard}>
          <p style={styles.cardLabel}>Avg ROM</p>
          <h3 style={styles.cardValue}>{Number(analysis.avg_rom || 0).toFixed(2)}</h3>
        </div>

        <div style={styles.summaryCard}>
          <p style={styles.cardLabel}>Avg Duration</p>
          <h3 style={styles.cardValue}>{Number(analysis.avg_duration || 0).toFixed(2)}s</h3>
        </div>
      </div>
    );
  }

  function renderArtifactLinks() {
    if (!selectedAnalysis) return null;

    const urls = selectedAnalysis.artifact_urls || {};

    const linkItems = [
      { key: "summary_json", label: "Open summary JSON" },
      { key: "reps_csv", label: "Open reps CSV" },
      { key: "angles_csv", label: "Open angle CSV" },
      { key: "reps_json", label: "Open reps JSON" },
      { key: "rep_metrics_json", label: "Open metrics JSON" },
    ];

    const visibleLinks = linkItems.filter((item) => urls[item.key]);

    if (visibleLinks.length === 0) {
      return <p style={styles.mutedText}>No extra artifact links available for this run.</p>;
    }

    return (
      <div style={styles.linkGrid}>
        {visibleLinks.map((item) => (
          <a
            key={item.key}
            href={getArtifactUrl(urls[item.key])}
            target="_blank"
            rel="noreferrer"
            style={styles.artifactLink}
          >
            {item.label}
          </a>
        ))}
      </div>
    );
  }

  function renderAnnotatedVideo() {
    if (!selectedAnalysis) return null;

    const videoUrl = getArtifactUrl(selectedAnalysis.artifact_urls?.annotated_video);

    if (!videoUrl) {
      return (
        <p style={styles.mutedText}>
          No annotated video was saved for this analysis. Turn on "Save annotated video" before running.
        </p>
      );
    }

    return (
      <div style={styles.videoShell}>
        <div style={styles.videoWrap}>
          <video controls style={styles.videoPlayer}>
            <source src={videoUrl} type="video/mp4" />
            Your browser does not support video playback.
          </video>
        </div>
      </div>
    );
  }

  function renderCoachPanel() {
    if (!selectedAnalysis) return null;

    const feedback = selectedAnalysis.feedback;

    return (
      <div style={styles.coachPanel}>
        <div>
          <p style={styles.coachEyebrow}>Coach Feedback</p>

          {!feedback ? (
            <>
              <h3 style={styles.coachTitle}>No feedback yet</h3>
              <p style={styles.coachSummary}>
                Run an analysis or load a saved session to view feedback.
              </p>
            </>
          ) : (
            <>
              <h3 style={styles.coachTitle}>{feedback.headline}</h3>
              <p style={styles.coachSummary}>{feedback.summary}</p>

              {feedback.bullets?.length > 0 && (
                <div style={styles.feedbackSection}>
                  <p style={styles.feedbackLabel}>Main takeaways</p>
                  <ul style={styles.feedbackList}>
                    {feedback.bullets.map((item, index) => (
                      <li key={index} style={styles.feedbackListItem}>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {feedback.highlights?.length > 0 && (
                <div style={styles.feedbackSection}>
                  <p style={styles.feedbackLabel}>Notes</p>
                  <ul style={styles.feedbackList}>
                    {feedback.highlights.map((item, index) => (
                      <li key={index} style={styles.feedbackListItem}>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>

        <div style={styles.chatBox}>
          <p style={styles.feedbackLabel}>Ask about this workout</p>

          <div style={styles.chatMessages}>
            {chatMessages.map((message, index) => (
              <div
                key={index}
                style={
                  message.role === "user"
                    ? styles.userMessageBubble
                    : styles.assistantMessageBubble
                }
              >
                {message.text}
              </div>
            ))}

            {chatLoading && (
              <div style={styles.assistantMessageBubble}>Thinking...</div>
            )}

            <div ref={chatEndRef} />
          </div>

          <div style={styles.chatInputRow}>
            <textarea
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={handleChatKeyDown}
              placeholder="Ask about ROM, tempo, best rep, failed reps, or set summary..."
              style={styles.chatInput}
              rows={3}
            />

            <button
              onClick={handleSendChat}
              disabled={chatLoading || !chatInput.trim() || !selectedAnalysis?.analysis?.id}
              style={styles.chatSendButton}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    );
  }

  function renderPlots() {
    if (!selectedAnalysis) return null;

    const urls = selectedAnalysis.artifact_urls || {};

    const plots = [
      { key: "joint_angles_plot", label: "Joint Angle Plot" },
      { key: "rep_timeline_plot", label: "Rep Timeline" },
      { key: "rep_rom_plot", label: "ROM per Rep" },
      { key: "rep_duration_plot", label: "Duration per Rep" },
      { key: "rep_outcomes_plot", label: "Rep Outcomes" },
    ].filter((plot) => urls[plot.key]);

    if (plots.length === 0) {
      return (
        <p style={styles.mutedText}>
          No plots were saved for this analysis. Turn on "Save plots" before running.
        </p>
      );
    }

    return (
      <div style={styles.plotGrid}>
        {plots.map((plot) => (
          <div key={plot.key} style={styles.plotCard}>
            <p style={styles.plotTitle}>{plot.label}</p>
            <img
              src={getArtifactUrl(urls[plot.key])}
              alt={plot.label}
              style={styles.plotImage}
            />
          </div>
        ))}
      </div>
    );
  }

  function renderRepTable() {
    if (!selectedAnalysis) return null;

    const reps = selectedAnalysis.reps || [];

    if (reps.length === 0) {
      return <p style={styles.mutedText}>No rep rows found for this analysis.</p>;
    }

    return (
      <div style={styles.tableWrap}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Rep</th>
              <th style={styles.th}>Start</th>
              <th style={styles.th}>End</th>
              <th style={styles.th}>Duration</th>
              <th style={styles.th}>ROM</th>
              <th style={styles.th}>Label</th>
              <th style={styles.th}>Reason</th>
            </tr>
          </thead>
          <tbody>
            {reps.map((rep) => {
              const isPass = String(rep.label || "").toLowerCase() === "pass";

              return (
                <tr key={rep.rep_index} style={isPass ? styles.passRow : styles.failRow}>
                  <td style={styles.td}>{rep.rep_index}</td>
                  <td style={styles.td}>{rep.start_idx}</td>
                  <td style={styles.td}>{rep.end_idx}</td>
                  <td style={styles.td}>{Number(rep.duration || 0).toFixed(2)}</td>
                  <td style={styles.td}>{Number(rep.rom || 0).toFixed(2)}</td>
                  <td style={styles.td}>
                    <span style={isPass ? styles.passBadge : styles.failBadge}>
                      {rep.label}
                    </span>
                  </td>
                  <td style={styles.td}>{rep.reason}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div style={styles.page}>
      <header style={styles.pageHeader}>
        <div>
          <p style={styles.eyebrow}>Gym Pose Estimation</p>
          <h1 style={styles.pageTitle}>Exercise Analysis Dashboard</h1>
          <p style={styles.subtext}>
            Upload a workout video, run the pipeline, and review saved sessions with video,
            plots, and rep-by-rep results.
          </p>
        </div>
      </header>

      <div style={styles.layout}>
        <aside style={styles.sidebar}>
          <div style={styles.panel}>
            <h2 style={styles.panelTitle}>Run Analysis</h2>

            <form onSubmit={handleSubmit} style={styles.form}>
              <label style={styles.fieldLabel}>
                <span style={styles.fieldText}>Exercise</span>
                <select
                  value={exercise}
                  onChange={(e) => setExercise(e.target.value)}
                  style={styles.input}
                >
                  <option value="curl">Curl</option>
                  <option value="bench">Bench</option>
                  <option value="squat">Squat</option>
                </select>
              </label>

              <label style={styles.fieldLabel}>
                <span style={styles.fieldText}>Video File</span>
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  style={styles.fileInput}
                />
              </label>

              <div style={styles.checkGrid}>
                <label style={styles.checkboxRow}>
                  <input
                    type="checkbox"
                    checked={saveVideo}
                    onChange={(e) => setSaveVideo(e.target.checked)}
                  />
                  <span>Save annotated video</span>
                </label>

                <label style={styles.checkboxRow}>
                  <input
                    type="checkbox"
                    checked={savePlots}
                    onChange={(e) => setSavePlots(e.target.checked)}
                  />
                  <span>Save plots</span>
                </label>

                <label style={styles.checkboxRow}>
                  <input
                    type="checkbox"
                    checked={saveAngleCsv}
                    onChange={(e) => setSaveAngleCsv(e.target.checked)}
                  />
                  <span>Save angle CSV</span>
                </label>

                <label style={styles.checkboxRow}>
                  <input
                    type="checkbox"
                    checked={saveRepsJson}
                    onChange={(e) => setSaveRepsJson(e.target.checked)}
                  />
                  <span>Save reps JSON</span>
                </label>
              </div>

              <button type="submit" disabled={loading || demoLoading} style={styles.submitButton}>
                {loading ? "Analyzing..." : "Analyze Video"}
              </button>

              <button
                type="button"
                onClick={handleRunDemo}
                disabled={loading || demoLoading}
                style={styles.demoButton}
              >
                {demoLoading ? "Running Demo..." : "Try Demo Curl"}
              </button>
            </form>

            {error && <p style={styles.errorText}>{error}</p>}
          </div>

          <div style={styles.panel}>
            <h2 style={styles.panelTitle}>Saved Analyses</h2>

            <div style={styles.historyList}>
              {analyses.length === 0 && (
                <p style={styles.mutedText}>No saved analyses yet.</p>
              )}

              {analyses.map((analysis) => {
                const isSelected = selectedAnalysis?.analysis?.id === analysis.id;

                return (
                  <button
                    key={analysis.id}
                    onClick={() => fetchAnalysisDetail(analysis.id)}
                    style={{
                      ...styles.historyItem,
                      ...(isSelected ? styles.historyItemSelected : {}),
                    }}
                  >
                    <div style={styles.historyTop}>
                      <strong>{formatExerciseName(analysis.exercise)}</strong>
                      <span style={styles.historyId}>#{analysis.id}</span>
                    </div>

                    <p style={styles.historyFile}>{analysis.original_filename}</p>

                    <small style={styles.historyMeta}>
                      reps: {analysis.rep_count} | pass: {analysis.pass_count} | fail: {analysis.fail_count}
                    </small>
                  </button>
                );
              })}
            </div>
          </div>
        </aside>

        <main style={styles.content}>
          {!selectedAnalysis && (
            <div style={styles.panel}>
              <h2 style={styles.panelTitle}>No analysis selected</h2>
              <p style={styles.mutedText}>
                Run a new analysis, click one from the saved list, or use the demo button.
              </p>
            </div>
          )}

          {selectedAnalysis && (
            <>
              <div style={styles.panel}>
                <h2 style={styles.panelTitle}>Session Summary</h2>
                {renderSummaryCards()}
              </div>

              <div style={styles.panel}>
                <h2 style={styles.panelTitle}>Session Details</h2>

                <div style={styles.detailGrid}>
                  <div>
                    <p style={styles.cardLabel}>Original File</p>
                    <p style={styles.detailValue}>{selectedAnalysis.analysis.original_filename}</p>
                  </div>

                  <div>
                    <p style={styles.cardLabel}>Created At</p>
                    <p style={styles.detailValue}>{selectedAnalysis.analysis.created_at}</p>
                  </div>

                  <div>
                    <p style={styles.cardLabel}>Output Directory</p>
                    <p style={styles.pathText}>{selectedAnalysis.analysis.output_dir}</p>
                  </div>
                </div>
              </div>

              <div style={styles.analysisSplit}>
                <div style={styles.panel}>
                  <h2 style={styles.panelTitle}>Annotated Video</h2>
                  {renderAnnotatedVideo()}
                </div>

                <div style={styles.panel}>
                  {renderCoachPanel()}
                </div>
              </div>

              <div style={styles.panel}>
                <h2 style={styles.panelTitle}>Plots</h2>
                {renderPlots()}
              </div>

              <div style={styles.panel}>
                <h2 style={styles.panelTitle}>Artifacts</h2>
                {renderArtifactLinks()}
              </div>

              <div style={styles.panel}>
                <h2 style={styles.panelTitle}>Rep Breakdown</h2>
                {renderRepTable()}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#0b1220",
    color: "#e5e7eb",
    fontFamily: "Inter, Arial, sans-serif",
    padding: "24px",
    boxSizing: "border-box",
  },
  pageHeader: {
    marginBottom: "24px",
  },
  eyebrow: {
    margin: 0,
    color: "#60a5fa",
    fontSize: "0.9rem",
    fontWeight: 700,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
  },
  pageTitle: {
    margin: "8px 0 10px 0",
    fontSize: "2rem",
    color: "#f8fafc",
  },
  subtext: {
    margin: 0,
    color: "#94a3b8",
    maxWidth: "760px",
    lineHeight: 1.6,
  },
  layout: {
    display: "grid",
    gridTemplateColumns: "360px 1fr",
    gap: "20px",
    alignItems: "start",
  },
  sidebar: {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  },
  content: {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  },
  panel: {
    background: "#111827",
    border: "1px solid #1f2937",
    borderRadius: "16px",
    padding: "20px",
    boxShadow: "0 10px 30px rgba(0, 0, 0, 0.22)",
  },
  panelTitle: {
    marginTop: 0,
    marginBottom: "16px",
    color: "#f8fafc",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  fieldLabel: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  fieldText: {
    fontSize: "0.92rem",
    color: "#cbd5e1",
    fontWeight: 600,
  },
  input: {
    background: "#0f172a",
    color: "#e5e7eb",
    border: "1px solid #334155",
    borderRadius: "10px",
    padding: "10px 12px",
    fontSize: "0.95rem",
  },
  fileInput: {
    background: "#0f172a",
    color: "#cbd5e1",
    border: "1px solid #334155",
    borderRadius: "10px",
    padding: "10px 12px",
  },
  checkGrid: {
    display: "grid",
    gridTemplateColumns: "1fr",
    gap: "10px",
    padding: "4px 0",
  },
  checkboxRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    color: "#cbd5e1",
    fontSize: "0.95rem",
  },
  submitButton: {
    background: "#2563eb",
    color: "#ffffff",
    border: "none",
    borderRadius: "10px",
    padding: "12px 14px",
    fontWeight: 700,
    cursor: "pointer",
    fontSize: "0.96rem",
  },
  demoButton: {
    background: "#0f172a",
    color: "#93c5fd",
    border: "1px solid #2563eb",
    borderRadius: "10px",
    padding: "12px 14px",
    fontWeight: 700,
    cursor: "pointer",
    fontSize: "0.96rem",
  },
  errorText: {
    marginTop: "14px",
    color: "#fca5a5",
    fontWeight: 600,
  },
  mutedText: {
    color: "#94a3b8",
    margin: 0,
    lineHeight: 1.6,
  },
  historyList: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },
  historyItem: {
    textAlign: "left",
    background: "#0f172a",
    color: "#e5e7eb",
    border: "1px solid #1e293b",
    borderRadius: "12px",
    padding: "14px",
    cursor: "pointer",
  },
  historyItemSelected: {
    border: "1px solid #3b82f6",
    background: "#111c33",
  },
  historyTop: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "8px",
  },
  historyId: {
    color: "#94a3b8",
    fontSize: "0.9rem",
  },
  historyFile: {
    margin: "0 0 6px 0",
    color: "#cbd5e1",
    wordBreak: "break-word",
  },
  historyMeta: {
    color: "#94a3b8",
  },
  summaryGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
    gap: "14px",
  },
  summaryCard: {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: "14px",
    padding: "16px",
  },
  cardLabel: {
    margin: "0 0 8px 0",
    color: "#94a3b8",
    fontSize: "0.84rem",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  },
  cardValue: {
    margin: 0,
    color: "#f8fafc",
    fontSize: "1.35rem",
  },
  detailGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
    gap: "16px",
  },
  detailValue: {
    margin: 0,
    color: "#e5e7eb",
    lineHeight: 1.5,
  },
  pathText: {
    margin: 0,
    color: "#cbd5e1",
    lineHeight: 1.5,
    wordBreak: "break-word",
  },
  analysisSplit: {
    display: "grid",
    gridTemplateColumns: "minmax(0, 1.15fr) minmax(320px, 0.85fr)",
    gap: "20px",
    alignItems: "stretch",
  },
  videoShell: {
    background: "#0a0f1c",
    border: "1px solid #1e293b",
    borderRadius: "16px",
    padding: "16px",
  },
  videoWrap: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    background: "#020617",
    borderRadius: "12px",
    overflow: "hidden",
    minHeight: "320px",
    maxHeight: "520px",
  },
  videoPlayer: {
    width: "100%",
    maxHeight: "520px",
    display: "block",
    background: "#000000",
    objectFit: "contain",
    borderRadius: "10px",
  },
  coachPanel: {
    height: "100%",
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  coachEyebrow: {
    margin: 0,
    color: "#60a5fa",
    fontSize: "0.85rem",
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  },
  coachTitle: {
    margin: 0,
    color: "#f8fafc",
    fontSize: "1.45rem",
  },
  coachSummary: {
    margin: 0,
    color: "#cbd5e1",
    lineHeight: 1.7,
  },
  feedbackSection: {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: "14px",
    padding: "14px",
  },
  feedbackLabel: {
    margin: "0 0 10px 0",
    color: "#93c5fd",
    fontWeight: 700,
    fontSize: "0.95rem",
  },
  feedbackList: {
    margin: 0,
    paddingLeft: "18px",
    color: "#dbeafe",
    lineHeight: 1.7,
  },
  feedbackListItem: {
    marginBottom: "8px",
  },
  chatBox: {
    marginTop: "auto",
    background: "linear-gradient(180deg, #101826 0%, #0f172a 100%)",
    border: "1px solid #1e3a8a",
    borderRadius: "14px",
    padding: "16px",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    minHeight: "320px",
  },
  chatMessages: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    background: "#08101d",
    border: "1px solid #1e293b",
    borderRadius: "12px",
    padding: "12px",
    minHeight: "170px",
    maxHeight: "260px",
    overflowY: "auto",
  },
  assistantMessageBubble: {
    alignSelf: "flex-start",
    background: "#162033",
    color: "#dbeafe",
    border: "1px solid #22314d",
    borderRadius: "12px",
    padding: "10px 12px",
    maxWidth: "95%",
    lineHeight: 1.6,
    whiteSpace: "pre-wrap",
  },
  userMessageBubble: {
    alignSelf: "flex-end",
    background: "#2563eb",
    color: "#ffffff",
    borderRadius: "12px",
    padding: "10px 12px",
    maxWidth: "95%",
    lineHeight: 1.6,
    whiteSpace: "pre-wrap",
  },
  chatInputRow: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },
  chatInput: {
    width: "100%",
    resize: "vertical",
    minHeight: "84px",
    background: "#0b1220",
    color: "#e5e7eb",
    border: "1px solid #334155",
    borderRadius: "10px",
    padding: "12px",
    fontSize: "0.95rem",
    fontFamily: "Inter, Arial, sans-serif",
    boxSizing: "border-box",
  },
  chatSendButton: {
    alignSelf: "flex-end",
    background: "#2563eb",
    color: "#ffffff",
    border: "none",
    borderRadius: "10px",
    padding: "10px 16px",
    fontWeight: 700,
    cursor: "pointer",
    fontSize: "0.95rem",
  },
  plotGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: "16px",
  },
  plotCard: {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: "14px",
    padding: "14px",
  },
  plotTitle: {
    marginTop: 0,
    marginBottom: "12px",
    color: "#e2e8f0",
    fontWeight: 700,
  },
  plotImage: {
    width: "100%",
    borderRadius: "10px",
    display: "block",
  },
  linkGrid: {
    display: "flex",
    flexWrap: "wrap",
    gap: "12px",
  },
  artifactLink: {
    background: "#0f172a",
    color: "#93c5fd",
    border: "1px solid #1e3a8a",
    borderRadius: "10px",
    padding: "10px 14px",
    textDecoration: "none",
    fontWeight: 600,
  },
  tableWrap: {
    overflowX: "auto",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    minWidth: "760px",
  },
  th: {
    textAlign: "left",
    padding: "12px",
    borderBottom: "1px solid #334155",
    color: "#cbd5e1",
    fontSize: "0.9rem",
  },
  td: {
    padding: "12px",
    borderBottom: "1px solid #1e293b",
    color: "#e5e7eb",
    verticalAlign: "top",
  },
  passRow: {
    background: "rgba(34, 197, 94, 0.05)",
  },
  failRow: {
    background: "rgba(239, 68, 68, 0.05)",
  },
  passBadge: {
    display: "inline-block",
    background: "rgba(34, 197, 94, 0.18)",
    color: "#86efac",
    border: "1px solid rgba(34, 197, 94, 0.35)",
    borderRadius: "999px",
    padding: "4px 10px",
    fontSize: "0.82rem",
    fontWeight: 700,
    textTransform: "capitalize",
  },
  failBadge: {
    display: "inline-block",
    background: "rgba(239, 68, 68, 0.18)",
    color: "#fca5a5",
    border: "1px solid rgba(239, 68, 68, 0.35)",
    borderRadius: "999px",
    padding: "4px 10px",
    fontSize: "0.82rem",
    fontWeight: 700,
    textTransform: "capitalize",
  },
};

export default App;
