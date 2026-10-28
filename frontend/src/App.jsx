import { useEffect, useState } from "react";
import {
  Activity,
  ClipboardCheck,
  History,
  ImagePlus,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import "./App.css";

const API = "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [metadata, setMetadata] = useState({
    machine_id: "M-01",
    tool_id: "T-01",
    batch_id: "B-001",
    shift: "Day",
    data_source: "real_sample",
  });
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [analytics, setAnalytics] = useState({
    total_inspections: 0,
    status_counts: [],
    batch_summary: [],
  });
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const [error, setError] = useState("");

  async function loadHistory() {
    try {
      const [historyResponse, analyticsResponse] = await Promise.all([
        fetch(`${API}/inspections`),
        fetch(`${API}/analytics`),
      ]);

      if (!historyResponse.ok || !analyticsResponse.ok) {
        throw new Error("Could not load dashboard data.");
      }

      const historyData = await historyResponse.json();
      const analyticsData = await analyticsResponse.json();

      setHistory(
        Array.isArray(historyData) ? historyData : (historyData.value ?? []),
      );
      setAnalytics(analyticsData);
      setLastRefreshed(new Date());
    } catch {
      setError("Cannot connect to the backend. Check that FastAPI is running.");
    }
  }

  async function loadAnalytics() {
    try {
      const response = await fetch(`${API}/analytics`);

      if (!response.ok) {
        throw new Error("Could not load analytics");
      }

      setAnalytics(await response.json());
    } catch (err) {
      console.error("Analytics loading failed:", err);
    }
  }

  useEffect(() => {
    loadHistory();
    loadAnalytics();
  }, []);

  function updateField(event) {
    setMetadata((previous) => ({
      ...previous,
      [event.target.name]: event.target.value,
    }));
  }

  async function submitInspection(event) {
    event.preventDefault();

    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    Object.entries(metadata).forEach(([key, value]) => {
      formData.append(key, value);
    });

    try {
      const response = await fetch(`${API}/inspect`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Inspection failed");
      }

      setResult(data);
      await Promise.all([loadHistory(), loadAnalytics()]);
    } catch (err) {
      setError(err.message || "Inspection failed. Please retry.");
    } finally {
      setLoading(false);
    }
  }

  const reviewCount = history.filter(
    (item) => item.inspection_status === "Review required",
  ).length;

  const productionAnalysis = (field) => {
    const groups = {};

    history.forEach((item) => {
      const key = item[field] || "Unknown";

      if (!groups[key]) {
        groups[key] = {
          name: key,
          total: 0,
          flagged: 0,
          pending: 0,
        };
      }

      groups[key].total += 1;

      if (item.inspection_status === "Review required") {
        groups[key].flagged += 1;
      }

      if (item.inspection_status === "Pending reference") {
        groups[key].pending += 1;
      }
    });

    return Object.values(groups).sort(
      (a, b) => b.flagged - a.flagged || b.total - a.total,
    );
  };

  const machineAnalysis = productionAnalysis("machine_id");
  const toolAnalysis = productionAnalysis("tool_id");
  const shiftAnalysis = productionAnalysis("shift");

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark">Q</div>
        <div>
          <h2>Q-Insight</h2>
          <p className="muted">Quality Intelligence</p>
        </div>

        <nav>
          <a href="#overview">
            <Activity size={18} /> Overview
          </a>
          <a href="#inspection">
            <ImagePlus size={18} /> New inspection
          </a>
          <a href="#history">
            <History size={18} /> History
          </a>
          <a href="#analytics">
            <Activity size={18} /> Analytics
          </a>
        </nav>

        <div className="sidebar-note">
          <ShieldCheck size={20} />
          <p>Visual inspection with production context</p>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">MANUFACTURING QUALITY</p>
            <h1>Inspection workspace</h1>
          </div>

          {lastRefreshed && (
            <p className="muted">
              History last refreshed at {lastRefreshed.toLocaleTimeString()}
            </p>
          )}

          <button
            className="secondary-button"
            onClick={() => Promise.all([loadHistory(), loadAnalytics()])}
          >
            <RefreshCw size={16} /> Refresh
          </button>
        </header>

        {lastRefreshed && (
          <p className="refresh-info">
            History last refreshed at {lastRefreshed.toLocaleTimeString()}
          </p>
        )}

        {error && <div className="error-banner">{error}</div>}

        <section id="overview" className="stats-grid">
          <article className="stat-card">
            <span>Total inspections</span>
            <strong>{analytics?.total_inspections ?? history.length}</strong>
            <small>Saved inspection records</small>
          </article>

          <article className="stat-card">
            <span>Flagged for review</span>
            <strong>
              {analytics?.status_counts?.find(
                (item) => item.inspection_status === "Review required",
              )?.count ?? reviewCount}
            </strong>
            <small>Require human verification</small>
          </article>

          <article className="stat-card">
            <span>Inspection engine</span>
            <strong className="engine-status">Online</strong>
            <small>OpenCV reference comparison</small>
          </article>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">QUALITY ANALYTICS</p>
              <h2>Batch performance</h2>
              <p className="muted">
                Inspection findings grouped by production batch.
              </p>
            </div>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Batch ID</th>
                  <th>Total inspections</th>
                  <th>Flagged for review</th>
                </tr>
              </thead>
              <tbody>
                {(analytics?.batch_summary ?? []).map((batch) => (
                  <tr key={batch.batch_id}>
                    <td>{batch.batch_id}</td>
                    <td>{batch.total}</td>
                    <td>{batch.flagged}</td>
                  </tr>
                ))}

                {!analytics?.batch_summary?.length && (
                  <tr>
                    <td colSpan="3">No batch analytics available yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section id="inspection" className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">01 / IMAGE ANALYSIS</p>
              <h2>New inspection</h2>
              <p className="muted">
                Upload an image and associate it with production metadata.
              </p>
            </div>
          </div>

          <form onSubmit={submitInspection}>
            <label className="upload-box">
              <ImagePlus size={30} />
              <strong>
                {file ? file.name : "Choose a switch-plate image"}
              </strong>
              <span>JPG, PNG or WEBP · Maximum 10 MB</span>
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event) => {
                  setFile(event.target.files?.[0] || null);
                  setResult(null);
                  setError("");
                }}
              />
            </label>

            {file && (
              <img
                className="image-preview"
                src={URL.createObjectURL(file)}
                alt="Selected inspection sample"
              />
            )}

            <div className="form-grid">
              {[
                ["machine_id", "Machine ID"],
                ["tool_id", "Tool ID"],
                ["batch_id", "Batch ID"],
              ].map(([name, label]) => (
                <label key={name}>
                  {label}
                  <input
                    name={name}
                    value={metadata[name]}
                    onChange={updateField}
                    required
                  />
                </label>
              ))}

              <label>
                Shift
                <select
                  name="shift"
                  value={metadata.shift}
                  onChange={updateField}
                >
                  <option>Day</option>
                  <option>Evening</option>
                  <option>Night</option>
                </select>
              </label>

              <label>
                Data source
                <select
                  name="data_source"
                  value={metadata.data_source}
                  onChange={updateField}
                >
                  <option value="real_sample">Real sample</option>
                  <option value="synthetic_demo">Synthetic demo</option>
                </select>
              </label>
            </div>

            <button className="primary-button" disabled={loading}>
              <ClipboardCheck size={18} />
              {loading ? "Inspecting..." : "Run inspection"}
            </button>
          </form>
        </section>

        {result && (
          <section className="panel result-panel">
            <p className="eyebrow">02 / INSPECTION RESULT</p>
            <h2>{result.defect_label}</h2>
            <span className="status-pill">{result.inspection_status}</span>

            <div className="result-grid">
              <div>
                <span>Anomaly score</span>
                <strong>
                  {result.anomaly_score == null
                    ? "N/A"
                    : Number(result.anomaly_score).toFixed(4)}
                </strong>
              </div>

              <div>
                <span>Similarity score</span>
                <strong>
                  {result.similarity_score == null
                    ? "N/A"
                    : Number(result.similarity_score).toFixed(4)}
                </strong>
              </div>
            </div>

            <h3>Explanation</h3>
            <p>{result.explanation}</p>

            <h3>Recommended next action</h3>
            <p>{result.recommended_action}</p>

            {/* NEW: Visual Difference Map */}
            {result.difference_map && (
              <div className="difference-map-section">
                <h3>Visual Difference Map</h3>

                <p className="muted">
                  Highlighted regions indicate visual differences between the
                  uploaded image and the configured reference image.
                </p>

                <img
                  src={result.difference_map}
                  alt="Visual difference heatmap comparing the inspection
                  image with the reference image"
                  className="difference-map-image"
                />

                <p className="muted">
                  Brightly highlighted areas represent detected pixel
                  differences, not confirmed manufacturing defects. Check image
                  alignment and lighting, then verify the result manually.
                </p>
              </div>
            )}

            {!result.difference_map &&
              result.inspection_status === "Pending reference" && (
                <div className="insight-note">
                  A reference image is not configured yet. Add a valid reference
                  image to enable visual difference analysis.
                </div>
              )}

            <p className="muted">
              Image-difference scores are not defect probabilities. Human review
              is required.
            </p>
          </section>
        )}

        <section id="analytics" className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">03 / QUALITY INTELLIGENCE</p>
              <h2>Quality analytics</h2>
              <p className="muted">
                Inspection trends and production-batch summaries.
              </p>
            </div>
          </div>

          <div className="stats-grid">
            {(analytics.status_counts ?? []).map((item) => (
              <article className="stat-card" key={item.inspection_status}>
                <span>{item.inspection_status}</span>
                <strong>{item.count}</strong>
                <small>Inspection records</small>
              </article>
            ))}
          </div>

          <h3>Batch performance</h3>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Batch ID</th>
                  <th>Total inspections</th>
                  <th>Flagged for review</th>
                  <th>Review rate</th>
                </tr>
              </thead>
              <tbody>
                {(analytics.batch_summary ?? []).map((batch) => (
                  <tr key={batch.batch_id}>
                    <td>{batch.batch_id}</td>
                    <td>{batch.total}</td>
                    <td>{batch.flagged}</td>
                    <td>
                      {batch.total > 0
                        ? `${((batch.flagged / batch.total) * 100).toFixed(1)}%`
                        : "N/A"}
                    </td>
                  </tr>
                ))}

                {(analytics.batch_summary ?? []).length === 0 && (
                  <tr>
                    <td colSpan="4">No batch analytics available yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section id="production-insights" className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">04 / PRODUCTION INTELLIGENCE</p>
              <h2>Production factor analysis</h2>
              <p className="muted">
                Compare inspection outcomes across machines, tools and shifts.
                These are associations, not proof of manufacturing causation.
              </p>
              <p className="insight-note">
                These results describe patterns in the available inspection
                records. They do not establish causation. Small or synthetic
                datasets may be misleading; verify findings with additional
                labelled samples and domain-expert review before taking
                corrective action.
              </p>
            </div>
          </div>

          {[
            { title: "Machine-wise analysis", rows: machineAnalysis },
            { title: "Tool-wise analysis", rows: toolAnalysis },
            { title: "Shift-wise analysis", rows: shiftAnalysis },
          ].map(({ title, rows }) => (
            <div className="production-analysis" key={title}>
              <h3>{title}</h3>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Production factor</th>
                      <th>Total</th>
                      <th>Review required</th>
                      <th>Pending reference</th>
                      <th>Review rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.name}>
                        <td>{row.name}</td>
                        <td>{row.total}</td>
                        <td>{row.flagged}</td>
                        <td>{row.pending}</td>
                        <td>
                          {row.total
                            ? `${((row.flagged / row.total) * 100).toFixed(1)}%`
                            : "N/A"}
                        </td>
                      </tr>
                    ))}

                    {rows.length === 0 && (
                      <tr>
                        <td colSpan="5">No inspection records available.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          ))}

          <div className="insight-note">
            <strong>Interpretation note</strong>
            <p>
              A higher review rate identifies a group worth investigating. It
              does not establish that the machine, tool or shift caused the
              difference. Small sample sizes, lighting and camera alignment can
              affect results.
            </p>
          </div>
        </section>

        <section id="history" className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">05 / RECORDS</p>
              <h2>Inspection history</h2>
            </div>
            <span className="record-count">{history.length} records</span>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Batch</th>
                  <th>Machine</th>
                  <th>Finding</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.id}>
                    <td>{new Date(item.timestamp).toLocaleString()}</td>
                    <td>{item.batch_id}</td>
                    <td>{item.machine_id}</td>
                    <td>{item.defect_label}</td>
                    <td>{item.inspection_status}</td>
                  </tr>
                ))}

                {history.length === 0 && (
                  <tr>
                    <td colSpan="5">
                      No inspections yet. Upload an image to begin.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <footer>Q-Insight · Prototype · Human verification required</footer>
      </section>
    </main>
  );
}
