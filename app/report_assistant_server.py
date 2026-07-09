import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT_PATH = ROOT / "app" / "report" / "taylor_swift_interactive_report.html"
FEEDBACK_PATH = ROOT / "logs" / "feedback.jsonl"

sys.path.append(str(SRC))

from router_local import answer_auto

app = FastAPI(title="Taylor Swift Music Intelligence")


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class FeedbackRequest(BaseModel):
    question: str
    feedback: str
    intent: str = ""
    answer: str = ""
    reliability: dict = {}


def dataframe_to_records(df):
    if isinstance(df, pd.DataFrame):
        return df.fillna("").to_dict(orient="records")
    return []


def make_widget_html():
    return """
<!-- Native Music Assistant -->
<style>
#music-assistant-toggle {
  position: fixed;
  right: 28px;
  bottom: 28px;
  z-index: 9999;
  border: none;
  border-radius: 999px;
  padding: 14px 20px;
  background: linear-gradient(135deg, #5a8fa8, #e8a0b4);
  color: white;
  font-weight: 700;
  font-size: 15px;
  box-shadow: 0 12px 30px rgba(90, 143, 168, 0.28);
  cursor: pointer;
}

#music-assistant-panel {
  position: fixed;
  right: 28px;
  bottom: 88px;
  width: min(480px, calc(100vw - 40px));
  height: min(720px, calc(100vh - 120px));
  background: #ffffff;
  border-radius: 22px;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 20px 60px rgba(0,0,0,0.18);
  overflow: hidden;
  z-index: 9998;
  display: none;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#music-assistant-header {
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 14px 0 18px;
  background: #f7f2eb;
  border-bottom: 1px solid rgba(0,0,0,0.06);
  font-weight: 700;
  color: #2c3a4a;
}

#music-assistant-close {
  border: none;
  background: transparent;
  font-size: 24px;
  cursor: pointer;
  color: #5a6a78;
}

#music-assistant-body {
  height: calc(100% - 50px);
  overflow-y: auto;
  padding: 16px;
}

#assistant-question {
  width: 100%;
  min-height: 82px;
  resize: vertical;
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 14px;
  padding: 12px;
  font-size: 14px;
  font-family: inherit;
  color: #2c3a4a;
  outline: none;
}

#assistant-question:focus {
  border-color: #5a8fa8;
  box-shadow: 0 0 0 3px rgba(90,143,168,0.12);
}

.assistant-actions {
  display: flex;
  gap: 10px;
  margin-top: 10px;
}

.assistant-actions button,
.assistant-example {
  border: 1px solid rgba(0,0,0,0.10);
  background: #ffffff;
  border-radius: 999px;
  padding: 9px 13px;
  cursor: pointer;
  font-weight: 600;
  color: #2c3a4a;
}

#assistant-ask {
  background: linear-gradient(135deg, #5a8fa8, #e8a0b4);
  color: white;
  border: none;
}

.assistant-examples {
  margin-top: 14px;
}

.assistant-example {
  width: 100%;
  text-align: left;
  margin-bottom: 8px;
  border-radius: 14px;
  font-size: 13px;
  font-weight: 500;
}

.assistant-section-title {
  margin-top: 16px;
  margin-bottom: 8px;
  font-weight: 800;
  color: #2c3a4a;
}

.assistant-muted {
  color: #7a8792;
  font-size: 13px;
}

.assistant-answer {
  white-space: pre-wrap;
  line-height: 1.7;
  color: #2c3a4a;
  background: #fbfaf8;
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 14px;
  padding: 12px;
}

.assistant-pill {
  display: inline-block;
  padding: 4px 9px;
  border-radius: 999px;
  background: #eef7ff;
  border: 1px solid #b8ddff;
  color: #125f9c;
  font-size: 12px;
  font-weight: 800;
}

.assistant-pill.analytics {
  background: #f4efff;
  border-color: #d8c5ff;
  color: #5b2ea6;
}

.assistant-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin-top: 8px;
}

.assistant-table th,
.assistant-table td {
  border-bottom: 1px solid rgba(0,0,0,0.06);
  padding: 7px 6px;
  text-align: left;
  vertical-align: top;
}

.assistant-table th {
  background: rgba(90,143,168,0.06);
  color: #2c3a4a;
}


.assistant-trace {
  margin-top: 8px;
  display: grid;
  gap: 8px;
}

.assistant-trace-item {
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(201,111,138,0.12);
}

.assistant-trace-step {
  font-weight: 800;
  color: #4b4052;
  font-size: 13px;
  margin-bottom: 4px;
}

.assistant-trace-detail {
  color: #6f6474;
  font-size: 13px;
  line-height: 1.55;
}


.assistant-answer-main {
  font-size: 14px !important;
  line-height: 1.65 !important;
  text-align: left !important;
}

.assistant-compact {
  font-size: 13px !important;
  line-height: 1.55 !important;
  text-align: left !important;
}

.assistant-reliability {
  display: grid;
  gap: 8px;
  padding: 12px;
  border-radius: 14px;
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(201,111,138,0.12);
  font-size: 13px;
  line-height: 1.5;
  color: #4b4052;
}

.assistant-details {
  margin-top: 14px;
  border-radius: 16px;
  border: 1px solid rgba(201,111,138,0.12);
  background: rgba(255,255,255,0.52);
  padding: 10px 12px;
}

.assistant-details summary {
  cursor: pointer;
  font-weight: 800;
  color: #4b4052;
  font-size: 13px;
}

#assistant-result,
#assistant-result * {
  text-align: left;
}

@media (max-width: 640px) {
  #music-assistant-panel {
    right: 12px;
    bottom: 78px;
    width: calc(100vw - 24px);
    height: calc(100vh - 100px);
  }

  #music-assistant-toggle {
    right: 16px;
    bottom: 18px;
  }
}
</style>

<button id="music-assistant-toggle">Ask Assistant</button>

<div id="music-assistant-panel">
  <div id="music-assistant-header">
    <span>Music Assistant</span>
    <button id="music-assistant-close">×</button>
  </div>

  <div id="music-assistant-body">
    <div class="assistant-muted">
      基于当前 Taylor Swift Spotify 数据集回答推荐和统计分析问题。
    </div>

    <textarea id="assistant-question">从 TTPD 里推荐几首 valence 低、energy 低的歌</textarea>

    <div class="assistant-actions">
      <button id="assistant-ask">Ask</button>
      <button id="assistant-clear">Clear</button>
    </div>

    <div class="assistant-examples">
      <div class="assistant-section-title">示例问题</div>
      <button class="assistant-example">从 TTPD 里推荐几首 valence 低、energy 低的歌</button>
      <button class="assistant-example">哪个专辑平均 valence 最低？</button>
      <button class="assistant-example">哪些歌 popularity 最高？</button>
      <button class="assistant-example">推荐几首适合运动的 high energy 歌</button>
    </div>

    <div id="assistant-result"></div>
  </div>
</div>

<script>
(function() {
  const toggle = document.getElementById("music-assistant-toggle");
  const panel = document.getElementById("music-assistant-panel");
  const close = document.getElementById("music-assistant-close");
  const ask = document.getElementById("assistant-ask");
  const clear = document.getElementById("assistant-clear");
  const question = document.getElementById("assistant-question");
  const result = document.getElementById("assistant-result");

  function esc(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function renderTable(rows) {
    if (!rows || rows.length === 0) return "";

    const keys = Object.keys(rows[0]).slice(0, 8);

    let html = '<table class="assistant-table"><thead><tr>';
    keys.forEach(k => html += `<th>${esc(k)}</th>`);
    html += '</tr></thead><tbody>';

    rows.forEach(row => {
      html += '<tr>';
      keys.forEach(k => html += `<td>${esc(row[k])}</td>`);
      html += '</tr>';
    });

    html += '</tbody></table>';
    return html;
  }

  function renderTrace(trace) {
    if (!trace || trace.length === 0) return "";

    let html = '<div class="assistant-trace">';
    trace.forEach(item => {
      html += `
        <div class="assistant-trace-item">
          <div class="assistant-trace-step">${esc(item.step)}</div>
          <div class="assistant-trace-detail">${esc(item.detail)}</div>
        </div>
      `;
    });
    html += '</div>';
    return html;
  }

  async function askAssistant() {
    const q = question.value.trim();
    if (!q) return;

    result.innerHTML = '<div class="assistant-section-title">Thinking...</div>';

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: q, top_k: 5})
      });

      const data = await resp.json();

      const intentClass = data.intent === "analytics" ? "assistant-pill analytics" : "assistant-pill";

      result.innerHTML = `
        <div class="assistant-section-title">Answer</div>
        <div class="assistant-answer assistant-answer-main">${esc(data.answer)}</div>

        <div class="assistant-section-title">${data.intent === "analytics" ? "Result Table" : "Evidence"}</div>
        ${renderTable(data.rows)}

        <div class="assistant-section-title">Reliability</div>
        <div class="assistant-reliability">
          <span class="${intentClass}">${esc(data.intent)}</span>
          <strong>${esc((data.reliability || {}).level || "Unknown")}</strong>
          <span>${esc((data.reliability || {}).reason || "")}</span>
        </div>

        <details class="assistant-details">
          <summary>How this answer was produced</summary>

          <div class="assistant-section-title">System Decision</div>
          <div class="assistant-answer assistant-compact">
            <strong>Reason:</strong> ${esc(data.route_reason || "")}<br>
            <strong>Pipeline:</strong> ${esc(data.pipeline || "")}
          </div>

          <div class="assistant-section-title">Query Trace</div>
          ${renderTrace(data.trace)}
        </details>
      `;
    } catch (err) {
      result.innerHTML = `<div class="assistant-answer">请求失败：${esc(err.message)}</div>`;
    }
  }

  toggle.addEventListener("click", function() {
    panel.style.display = panel.style.display === "block" ? "none" : "block";
  });

  close.addEventListener("click", function() {
    panel.style.display = "none";
  });

  ask.addEventListener("click", askAssistant);

  clear.addEventListener("click", function() {
    question.value = "";
    result.innerHTML = "";
  });

  document.querySelectorAll(".assistant-example").forEach(btn => {
    btn.addEventListener("click", function() {
      question.value = btn.textContent.trim();
    });
  });
})();
</script>
<!-- End Native Music Assistant -->
"""






def make_demo_focus_layout_html():
    return r"""
<!-- Demo Focus Layout -->
<style>
html {
  scroll-behavior: smooth;
}

/* 隐藏之前临时加到 header 里的单独 AI Studio 标签，统一使用新顶部导航 */
#ai-studio-nav-link {
  display: none !important;
}

/* 顶部导航 */
#demo-top-nav {
  position: sticky;
  top: 0;
  z-index: 9990;
  max-width: 1120px;
  margin: 18px auto 36px auto;
  padding: 0 28px;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
}

.demo-top-nav-inner {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  padding: 12px;
  border-radius: 999px;
  border: 1px solid rgba(201,111,138,0.14);
  background: rgba(255, 252, 248, 0.72);
  backdrop-filter: blur(14px);
  box-shadow:
    0 16px 48px rgba(88, 67, 96, 0.10),
    inset 0 1px 0 rgba(255,255,255,0.78);
}

.demo-top-nav-inner a {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 9px 14px;
  border-radius: 999px;
  text-decoration: none;
  color: #4b4052;
  font-size: 0.86rem;
  font-weight: 900;
  letter-spacing: 0.01em;
  border: 1px solid transparent;
}

.demo-top-nav-inner a:hover {
  background: linear-gradient(135deg, rgba(243,198,211,0.56), rgba(200,183,232,0.46), rgba(169,204,224,0.46));
  border-color: rgba(255,255,255,0.72);
}


/* 原生顶部导航增强：加入 AI Studio / 更多分析 / System Card */
.demo-native-top-nav .demo-native-extra-link {
  display: inline-flex !important;
  align-items: center;
  justify-content: center;
  margin-left: 12px;
  padding: 8px 14px;
  border-radius: 999px;
  text-decoration: none;
  color: #4b4052 !important;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
  font-size: 0.95em;
  font-weight: 800;
  letter-spacing: 0.01em;
  background: linear-gradient(135deg, rgba(243,198,211,0.66), rgba(200,183,232,0.56), rgba(169,204,224,0.56));
  border: 1px solid rgba(255,255,255,0.72);
  box-shadow:
    0 10px 24px rgba(150,113,150,0.12),
    inset 0 1px 0 rgba(255,255,255,0.76);
}

.demo-native-top-nav .demo-native-extra-link:hover {
  transform: translateY(-1px);
}

/* More Analysis 总容器 */
#more-analysis {
  max-width: 1220px;
  margin: 90px auto 96px auto;
  padding: 0 28px;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
}

.more-analysis-shell {
  border-radius: 34px;
  border: 1px solid rgba(201,111,138,0.16);
  background:
    radial-gradient(circle at 8% 0%, rgba(243,198,211,0.18), transparent 34%),
    radial-gradient(circle at 92% 16%, rgba(169,204,224,0.18), transparent 30%),
    linear-gradient(135deg, rgba(255,255,255,0.86), rgba(255,248,252,0.78));
  box-shadow:
    0 24px 76px rgba(88, 67, 96, 0.10),
    inset 0 1px 0 rgba(255,255,255,0.82);
  padding: 32px;
}

.more-analysis-eyebrow {
  color: #c96f8a;
  font-size: 0.78rem;
  font-weight: 900;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 10px;
}

#more-analysis h2 {
  margin: 0;
  font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
  color: #4b4052;
  font-size: clamp(2rem, 3vw, 3.4rem);
  line-height: 1;
  letter-spacing: -0.05em;
}

.more-analysis-desc {
  max-width: 860px;
  margin: 14px 0 26px 0;
  color: #6f6474;
  line-height: 1.7;
}

.more-analysis-list {
  display: grid;
  gap: 20px;
}

.more-analysis-card {
  border-radius: 26px;
  border: 1px solid rgba(201,111,138,0.12);
  background: rgba(255,255,255,0.60);
  overflow: hidden;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
}

.more-analysis-card-title {
  padding: 16px 20px;
  color: #4b4052;
  font-weight: 900;
  border-bottom: 1px solid rgba(201,111,138,0.10);
  background: rgba(255,255,255,0.46);
}

/* 被压缩的分析 section */
.demo-secondary-section {
  position: relative;
  max-height: 300px;
  overflow: hidden;
  transition: max-height 0.35s ease;
  padding: 18px 20px;
}

.demo-secondary-section.demo-expanded {
  max-height: none;
}

.demo-secondary-section::after {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 96px;
  pointer-events: none;
  background: linear-gradient(180deg, rgba(255,250,244,0), rgba(255,250,244,0.96));
}

.demo-secondary-section.demo-expanded::after {
  display: none;
}

.demo-expand-control {
  display: flex;
  justify-content: center;
  padding: 0 0 20px 0;
  position: relative;
  z-index: 5;
}

.demo-expand-control button {
  border: 1px solid rgba(201,111,138,0.18);
  border-radius: 999px;
  padding: 10px 18px;
  background: linear-gradient(135deg, #f3c6d3, #c8b7e8, #a9cce0);
  color: #332838;
  font-weight: 900;
  cursor: pointer;
  box-shadow:
    0 14px 34px rgba(150,113,150,0.18),
    inset 0 1px 0 rgba(255,255,255,0.76);
}

@media (max-width: 760px) {
  #demo-top-nav {
    position: static;
  }

  .demo-top-nav-inner {
    border-radius: 24px;
  }

  .more-analysis-shell {
    padding: 22px;
  }
}
</style>

<script>
document.addEventListener("DOMContentLoaded", function () {
  function textOf(el) {
    return (el && el.textContent ? el.textContent : "").trim();
  }

  function findSectionByKeywords(keywords) {
    const headings = Array.from(document.querySelectorAll("h1, h2, h3, .section-title"));

    for (const heading of headings) {
      const text = textOf(heading);
      const matched = keywords.some(keyword => text.includes(keyword));
      if (!matched) continue;

      const section =
        heading.closest("section") ||
        heading.closest(".section") ||
        heading.closest(".card") ||
        heading.closest(".panel") ||
        heading.parentElement;

      if (section && section !== document.body) return section;
    }

    return null;
  }

  function assignAnchor(id, keywords) {
    const section = findSectionByKeywords(keywords);
    if (section && !section.id) section.id = id;
    return section ? section.id : null;
  }

  function clearOldFocusControls() {
    document.querySelectorAll(".demo-expand-control, #demo-focus-note, #demo-top-nav, #more-analysis").forEach(el => {
      el.remove();
    });

    document.querySelectorAll(".demo-secondary-section").forEach(section => {
      section.classList.remove("demo-secondary-section", "demo-expanded");
      section.style.maxHeight = "";
    });
  }

  function collectSecondarySections() {
    const secondaryOrder = [
      {
        title: "聚类分析",
        keywords: ["聚类分析", "Clustering", "Cluster"]
      },
      {
        title: "流行度预测",
        keywords: ["流行度预测", "Popularity Prediction", "Prediction"]
      },
      {
        title: "Era 分类",
        keywords: ["Era 分类", "Era分类", "Era Classification"]
      },
      {
        title: "姊妹专辑",
        keywords: ["姊妹专辑", "Sister Albums", "Sister Album"]
      },
      {
        title: "Taylor\'s Version",
        keywords: ["Taylor’s Version", "Taylor's Version", "Taylor Version"]
      }
    ];

    const seen = new Set();
    const matched = [];

    secondaryOrder.forEach((item, index) => {
      const section = findSectionByKeywords(item.keywords);

      if (!section) return;
      if (section.id === "ai-studio" || section.id === "system-card") return;
      if (seen.has(section)) return;

      seen.add(section);
      matched.push({
        section,
        title: item.title,
        order: index
      });
    });

    return matched.sort((a, b) => a.order - b.order);
  }

  function buildMoreAnalysis(sections) {
    if (!sections || sections.length === 0) return;

    const aiStudio = document.getElementById("ai-studio");
    const systemCard = document.getElementById("system-card");

    const container = document.createElement("section");
    container.id = "more-analysis";
    container.innerHTML = `
      <div class="more-analysis-shell">
        <div class="more-analysis-eyebrow">More Analysis</div>
        <h2>附加探索分析</h2>
        <p class="more-analysis-desc">
          这些探索性分析仍然保留，但默认压缩展示，让主线更聚焦于数据叙事和 AI Music Intelligence Studio。
        </p>
        <div class="more-analysis-list"></div>
      </div>
    `;

    const list = container.querySelector(".more-analysis-list");

    sections.forEach(item => {
      const section = item.section;

      section.classList.add("demo-secondary-section");
      section.classList.remove("demo-expanded");

      const card = document.createElement("div");
      card.className = "more-analysis-card";

      const title = document.createElement("div");
      title.className = "more-analysis-card-title";
      title.textContent = item.title;

      const control = document.createElement("div");
      control.className = "demo-expand-control";
      control.innerHTML = '<button type="button">Show full analysis</button>';

      const button = control.querySelector("button");

      button.addEventListener("click", function () {
        section.classList.toggle("demo-expanded");
        button.textContent = section.classList.contains("demo-expanded")
          ? "Collapse analysis"
          : "Show full analysis";

        if (window.Plotly) {
          setTimeout(function () {
            section.querySelectorAll(".js-plotly-plot").forEach(function (plot) {
              try { window.Plotly.Plots.resize(plot); } catch (e) {}
            });
          }, 160);
        }
      });

      card.appendChild(title);
      card.appendChild(section);
      card.appendChild(control);
      list.appendChild(card);
    });

    if (aiStudio && aiStudio.parentNode) {
      aiStudio.insertAdjacentElement("afterend", container);
    } else if (systemCard && systemCard.parentNode) {
      systemCard.insertAdjacentElement("beforebegin", container);
    } else {
      document.body.appendChild(container);
    }
  }

  function buildTopNav() {
    // 移除之前动态生成的额外导航，避免重复
    document.querySelectorAll("#demo-top-nav").forEach(el => el.remove());

    const clickable = Array.from(document.querySelectorAll("a, button"));
    const overviewLink = clickable.find(el => {
      const t = textOf(el);
      return t.includes("数据概览") || t.includes("Overview");
    });

    let nav = overviewLink ? overviewLink.parentElement : null;

    // 如果找不到原导航，才创建一个兜底导航
    if (!nav) {
      const navItems = [
        ["数据概览", "overview"],
        ["音频特征", "audio-features"],
        ["风格演变", "era-evolution"],
        ["AI Studio", "ai-studio"],
        ["更多分析", "more-analysis"],
        ["System Card", "system-card"]
      ].filter(item => document.getElementById(item[1]));

      if (navItems.length === 0) return;

      const fallback = document.createElement("div");
      fallback.id = "demo-top-nav";
      fallback.innerHTML = `
        <div class="demo-top-nav-inner">
          ${navItems.map(([label, id]) => `<a href="#${id}">${label}</a>`).join("")}
        </div>
      `;

      const hero =
        document.querySelector("header") ||
        document.querySelector(".hero") ||
        document.querySelector(".cover") ||
        document.querySelector(".landing");

      if (hero && hero.parentNode) {
        hero.insertAdjacentElement("afterend", fallback);
      } else {
        document.body.insertAdjacentElement("afterbegin", fallback);
      }

      return;
    }

    nav.classList.add("demo-native-top-nav");

    // 隐藏原来不想放在主导航里的探索性标签
    const hideKeywords = [
      "聚类分析",
      "流行度预测",
      "Era 分类",
      "Era分类",
      "姊妹专辑",
      "Taylor's Version",
      "Taylor’s Version"
    ];

    nav.querySelectorAll("a, button").forEach(el => {
      const t = textOf(el);
      if (hideKeywords.some(k => t.includes(k))) {
        el.style.display = "none";
      }
    });

    // 删除旧的动态标签，防止重复
    nav.querySelectorAll(".demo-native-extra-link").forEach(el => el.remove());

    function makeLink(id, label) {
      if (!document.getElementById(id)) return null;

      const a = document.createElement("a");
      a.href = "#" + id;
      a.textContent = label;
      a.className = "demo-native-extra-link";
      return a;
    }

    const newLinks = [
      makeLink("ai-studio", "AI Studio"),
      makeLink("more-analysis", "更多分析"),
      makeLink("system-card", "System Card")
    ].filter(Boolean);

    const visibleLinks = Array.from(nav.querySelectorAll("a, button"))
      .filter(el => el.style.display !== "none");

    let anchor =
      visibleLinks.find(el => textOf(el).includes("风格演变")) ||
      visibleLinks[visibleLinks.length - 1];

    if (!anchor) {
      newLinks.forEach(link => nav.appendChild(link));
      return;
    }

    newLinks.forEach(link => {
      anchor.insertAdjacentElement("afterend", link);
      anchor = link;
    });
  }

  function prepareAnchors() {
    assignAnchor("overview", ["数据概览", "Dataset", "Overview", "概览"]);
    assignAnchor("audio-features", ["音频特征", "Audio Feature", "Audio Features"]);
    assignAnchor("era-evolution", ["风格演变", "Era Evolution", "Evolution", "演变"]);

    const ai = document.getElementById("ai-studio");
    const system = document.getElementById("system-card");

    if (ai) ai.id = "ai-studio";
    if (system) system.id = "system-card";
  }

  function removeOldSingleStudioLinkLater() {
    setTimeout(function () {
      const oldLink = document.getElementById("ai-studio-nav-link");
      if (oldLink) oldLink.remove();
    }, 500);
  }

  clearOldFocusControls();
  prepareAnchors();

  const secondarySections = collectSecondarySections();
  buildMoreAnalysis(secondarySections);
  buildTopNav();
  removeOldSingleStudioLinkLater();
});
</script>
<!-- End Demo Focus Layout -->
"""


def make_ai_studio_html():
    return r"""
<!-- AI Music Intelligence Studio -->
<section id="ai-studio">
  <div class="studio-shell">
    <div class="studio-header">
      <div>
        <div class="studio-eyebrow">AI Music Intelligence Studio</div>
        <h2>Ask the music data, not just the model.</h2>
        <p>
          Use Hybrid RAG for recommendation questions and deterministic pandas analytics
          for ranking, comparison, and feature statistics.
        </p>
      </div>
      <div class="studio-badge">Local Qwen · RAG · Analytics</div>
    </div>

    <div class="studio-grid">
      <div class="studio-control">
        <label for="studio-question">Question</label>
        <textarea id="studio-question">从 TTPD 里推荐几首 valence 低、energy 低的歌</textarea>

        <div class="studio-actions">
          <button id="studio-ask">Ask Studio</button>
          <button id="studio-clear">Clear</button>
        </div>

        <div class="studio-examples">
          <div class="studio-mini-title">Example Questions</div>
          <button class="studio-example">哪个专辑平均 valence 最低？</button>
          <button class="studio-example">从 TTPD 里推荐几首 valence 低、energy 低的歌</button>
          <button class="studio-example">TTPD 和 folklore 的 acousticness 有什么差异？</button>
          <button class="studio-example">推荐几首适合运动的 high energy 歌</button>
        </div>

        <div class="studio-history">
          <div class="studio-mini-title">Recent Questions</div>
          <div id="studio-history-list"></div>
        </div>
      </div>

      <div class="studio-output" id="studio-output">
        <div class="studio-placeholder">
          <div class="studio-placeholder-title">Ready for analysis.</div>
          <p>
            Ask a recommendation, ranking, comparison, or audio-feature question.
            The answer will appear here with evidence, reliability, and trace details.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>

<style>
#ai-studio {
  max-width: 1220px;
  margin: 88px auto 96px auto;
  padding: 0 28px;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
}

.studio-shell {
  border-radius: 36px;
  border: 1px solid rgba(201,111,138,0.16);
  background:
    radial-gradient(circle at 10% 0%, rgba(243,198,211,0.24), transparent 32%),
    radial-gradient(circle at 92% 20%, rgba(169,204,224,0.22), transparent 30%),
    linear-gradient(135deg, rgba(255,255,255,0.90), rgba(255,247,251,0.84));
  box-shadow:
    0 28px 90px rgba(88, 67, 96, 0.12),
    inset 0 1px 0 rgba(255,255,255,0.82);
  overflow: hidden;
}

.studio-header {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: flex-start;
  padding: 34px 38px 24px 38px;
  border-bottom: 1px solid rgba(201,111,138,0.10);
}

.studio-eyebrow {
  color: #c96f8a;
  font-size: 0.78rem;
  font-weight: 900;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 10px;
}

#ai-studio h2 {
  margin: 0;
  font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
  color: #4b4052;
  font-size: clamp(2.1rem, 3.2vw, 3.7rem);
  line-height: 1;
  letter-spacing: -0.055em;
}

.studio-header p {
  max-width: 720px;
  margin: 14px 0 0 0;
  color: #6f6474;
  line-height: 1.7;
  font-size: 0.98rem;
}

.studio-badge {
  white-space: nowrap;
  padding: 10px 14px;
  border-radius: 999px;
  color: #4b4052;
  font-size: 0.82rem;
  font-weight: 800;
  background: linear-gradient(135deg, rgba(243,198,211,0.52), rgba(200,183,232,0.42), rgba(169,204,224,0.42));
  border: 1px solid rgba(255,255,255,0.72);
}

.studio-grid {
  display: grid;
  grid-template-columns: minmax(300px, 0.85fr) minmax(420px, 1.35fr);
  gap: 24px;
  padding: 28px 38px 38px 38px;
}

.studio-control,
.studio-output {
  border-radius: 26px;
  border: 1px solid rgba(201,111,138,0.12);
  background: rgba(255,255,255,0.66);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.72);
}

.studio-control {
  padding: 22px;
}

.studio-output {
  min-height: 540px;
  padding: 24px;
  overflow: auto;
}

.studio-control label,
.studio-mini-title {
  display: block;
  color: #4b4052;
  font-weight: 900;
  font-size: 0.9rem;
  margin-bottom: 9px;
}

#studio-question {
  width: 100%;
  min-height: 128px;
  box-sizing: border-box;
  resize: vertical;
  border-radius: 20px;
  border: 1px solid rgba(201,111,138,0.16);
  background: rgba(255,255,255,0.74);
  color: #2f2734;
  padding: 14px 15px;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
  font-size: 0.96rem;
  line-height: 1.55;
  outline: none;
}

#studio-question:focus {
  border-color: rgba(111,145,172,0.55);
  box-shadow: 0 0 0 4px rgba(111,145,172,0.12);
}

.studio-actions {
  display: flex;
  gap: 10px;
  margin-top: 12px;
}

.studio-actions button,
.studio-example,
.studio-history-item,
.studio-feedback button {
  border: 1px solid rgba(201,111,138,0.16);
  border-radius: 999px;
  padding: 10px 14px;
  cursor: pointer;
  color: #332838;
  font-weight: 800;
  background: rgba(255,255,255,0.76);
}

#studio-ask {
  background: linear-gradient(135deg, #f3c6d3, #c8b7e8, #a9cce0);
  border-color: rgba(255,255,255,0.75);
}

.studio-examples,
.studio-history {
  margin-top: 22px;
}

.studio-example,
.studio-history-item {
  display: block;
  width: 100%;
  text-align: left;
  border-radius: 16px;
  margin-bottom: 8px;
  font-size: 0.9rem;
  line-height: 1.35;
  font-weight: 700;
}

.studio-placeholder {
  display: grid;
  place-content: center;
  min-height: 480px;
  text-align: center;
  color: #6f6474;
}

.studio-placeholder-title {
  color: #4b4052;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  margin-bottom: 10px;
}

.studio-section-title {
  color: #4b4052;
  font-weight: 900;
  font-size: 1rem;
  margin: 18px 0 10px 0;
}

.studio-answer,
.studio-reliability,
.studio-trace-item {
  border-radius: 20px;
  border: 1px solid rgba(201,111,138,0.12);
  background: rgba(255,255,255,0.72);
  padding: 15px;
  color: #2f2734;
  line-height: 1.72;
}

.studio-answer {
  white-space: pre-wrap;
  font-size: 0.98rem;
}

.studio-table-wrap {
  overflow-x: auto;
  border-radius: 20px;
  border: 1px solid rgba(201,111,138,0.12);
  background: rgba(255,255,255,0.68);
}

.studio-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}

.studio-table th,
.studio-table td {
  padding: 9px 10px;
  border-bottom: 1px solid rgba(201,111,138,0.10);
  text-align: left;
  vertical-align: top;
  color: #3b3140;
}

.studio-table th {
  color: #4b4052;
  background: rgba(243,198,211,0.22);
  font-weight: 900;
}

.studio-pill {
  display: inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(243,198,211,0.30);
  border: 1px solid rgba(201,111,138,0.22);
  color: #9d4f68;
  font-size: 0.78rem;
  font-weight: 900;
  margin-right: 8px;
}

.studio-pill.analytics {
  background: rgba(200,183,232,0.30);
  color: #5b4c93;
}

.studio-details {
  margin-top: 16px;
  border-radius: 20px;
  border: 1px solid rgba(201,111,138,0.12);
  background: rgba(255,255,255,0.48);
  padding: 13px 15px;
}

.studio-details summary {
  cursor: pointer;
  color: #4b4052;
  font-weight: 900;
}

.studio-trace {
  display: grid;
  gap: 9px;
  margin-top: 12px;
}

.studio-trace-step {
  color: #4b4052;
  font-weight: 900;
  margin-bottom: 4px;
}

.studio-trace-detail {
  color: #6f6474;
  font-size: 0.9rem;
  line-height: 1.55;
}

.studio-feedback {
  margin-top: 16px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}

.studio-feedback-label {
  color: #6f6474;
  font-weight: 800;
  font-size: 0.86rem;
}

.studio-feedback-status {
  color: #6f91ac;
  font-weight: 800;
  font-size: 0.86rem;
}

/* 右下角弹窗现在只是 Quick Ask，不再抢主角 */
#music-assistant-toggle::before {
  content: "" !important;
}


/* 顶部 AI Studio 入口 */
#ai-studio-nav-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-left: 14px;
  padding: 8px 14px;
  border-radius: 999px;
  text-decoration: none;
  color: #332838 !important;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
  font-size: 0.86rem;
  font-weight: 900;
  letter-spacing: 0.01em;
  background: linear-gradient(135deg, rgba(243,198,211,0.82), rgba(200,183,232,0.72), rgba(169,204,224,0.72));
  border: 1px solid rgba(255,255,255,0.72);
  box-shadow:
    0 10px 24px rgba(150,113,150,0.14),
    inset 0 1px 0 rgba(255,255,255,0.76);
}

#ai-studio-nav-link:hover {
  transform: translateY(-1px);
  box-shadow:
    0 14px 30px rgba(150,113,150,0.18),
    inset 0 1px 0 rgba(255,255,255,0.82);
}

/* 如果原页面没有可用 nav，则用一个轻量固定入口 */
#ai-studio-floating-nav {
  position: fixed;
  top: 18px;
  right: 24px;
  z-index: 9997;
}

html {
  scroll-behavior: smooth;
}


@media (max-width: 940px) {
  .studio-header {
    display: block;
  }

  .studio-badge {
    display: inline-block;
    margin-top: 16px;
  }

  .studio-grid {
    grid-template-columns: 1fr;
  }
}
</style>

<script>
(function () {
  const HISTORY_KEY = "taylor_ai_studio_history_v1";
  let lastStudioPayload = null;

  function esc(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function getHistory() {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    } catch {
      return [];
    }
  }

  function saveHistory(q) {
    q = String(q || "").trim();
    if (!q) return;

    let history = getHistory().filter(item => item !== q);
    history.unshift(q);
    history = history.slice(0, 6);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistory();
  }

  function renderHistory() {
    const host = document.getElementById("studio-history-list");
    if (!host) return;

    const history = getHistory();

    if (history.length === 0) {
      host.innerHTML = '<div style="color:#8b8190;font-size:0.88rem;">No history yet.</div>';
      return;
    }

    host.innerHTML = history.map(q =>
      `<button class="studio-history-item" data-question="${esc(q)}">${esc(q)}</button>`
    ).join("");

    host.querySelectorAll(".studio-history-item").forEach(btn => {
      btn.addEventListener("click", function () {
        const input = document.getElementById("studio-question");
        if (input) input.value = btn.dataset.question || "";
      });
    });
  }

  function renderTable(rows) {
    if (!rows || rows.length === 0) return '<div class="studio-answer">No evidence rows returned.</div>';

    const keys = Object.keys(rows[0]).slice(0, 8);

    let html = '<div class="studio-table-wrap"><table class="studio-table"><thead><tr>';
    keys.forEach(k => html += `<th>${esc(k)}</th>`);
    html += '</tr></thead><tbody>';

    rows.forEach(row => {
      html += '<tr>';
      keys.forEach(k => html += `<td>${esc(row[k])}</td>`);
      html += '</tr>';
    });

    html += '</tbody></table></div>';
    return html;
  }

  function renderTrace(trace) {
    if (!trace || trace.length === 0) return "";

    let html = '<div class="studio-trace">';
    trace.forEach(item => {
      html += `
        <div class="studio-trace-item">
          <div class="studio-trace-step">${esc(item.step)}</div>
          <div class="studio-trace-detail">${esc(item.detail)}</div>
        </div>
      `;
    });
    html += '</div>';
    return html;
  }

  async function askStudio() {
    const input = document.getElementById("studio-question");
    const output = document.getElementById("studio-output");
    const q = input ? input.value.trim() : "";

    if (!q || !output) return;

    output.innerHTML = '<div class="studio-placeholder"><div class="studio-placeholder-title">Thinking...</div><p>Routing query and preparing grounded answer.</p></div>';

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: q, top_k: 5})
      });

      const data = await resp.json();
      const intentClass = data.intent === "analytics" ? "studio-pill analytics" : "studio-pill";
      const tableTitle = data.intent === "analytics" ? "Result Table" : "Evidence";

      lastStudioPayload = {
        question: q,
        feedback: "",
        intent: data.intent || "",
        answer: data.answer || "",
        reliability: data.reliability || {}
      };

      saveHistory(q);

      output.innerHTML = `
        <div class="studio-section-title">Answer</div>
        <div class="studio-answer">${esc(data.answer)}</div>

        <div class="studio-section-title">${tableTitle}</div>
        ${renderTable(data.rows)}

        <div class="studio-section-title">Reliability</div>
        <div class="studio-reliability">
          <span class="${intentClass}">${esc(data.intent)}</span>
          <strong>${esc((data.reliability || {}).level || "Unknown")}</strong>
          <div style="margin-top:8px;">${esc((data.reliability || {}).reason || "")}</div>
        </div>

        <details class="studio-details">
          <summary>How this answer was produced</summary>

          <div class="studio-section-title">System Decision</div>
          <div class="studio-answer">
            <strong>Reason:</strong> ${esc(data.route_reason || "")}<br>
            <strong>Pipeline:</strong> ${esc(data.pipeline || "")}
          </div>

          <div class="studio-section-title">Query Trace</div>
          ${renderTrace(data.trace)}
        </details>

        <div class="studio-feedback">
          <span class="studio-feedback-label">Was this useful?</span>
          <button data-studio-feedback="up">👍 Helpful</button>
          <button data-studio-feedback="down">👎 Not Helpful</button>
          <span id="studio-feedback-status" class="studio-feedback-status"></span>
        </div>
      `;

      output.querySelectorAll("[data-studio-feedback]").forEach(btn => {
        btn.addEventListener("click", function () {
          sendStudioFeedback(btn.dataset.studioFeedback);
        });
      });

    } catch (err) {
      output.innerHTML = `<div class="studio-answer">Request failed: ${esc(err.message)}</div>`;
    }
  }

  async function sendStudioFeedback(value) {
    const status = document.getElementById("studio-feedback-status");
    if (!lastStudioPayload) return;

    const payload = {
      ...lastStudioPayload,
      feedback: value
    };

    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });

      if (status) status.textContent = "Saved.";
    } catch {
      if (status) status.textContent = "Failed to save.";
    }
  }

  function placeStudioLater() {
    const studio = document.getElementById("ai-studio");
    const systemCard = document.getElementById("system-card");

    if (!studio) return;

    // 默认把 AI Studio 放在正文报告之后、System Card 之前。
    if (systemCard && systemCard.parentNode && systemCard.previousElementSibling !== studio) {
      systemCard.insertAdjacentElement("beforebegin", studio);
    }
  }

  function addStudioTopNav() {
    if (document.getElementById("ai-studio-nav-link")) return;

    const link = document.createElement("a");
    link.id = "ai-studio-nav-link";
    link.href = "#ai-studio";
    link.textContent = "AI Studio";

    const nav =
      document.querySelector("nav") ||
      document.querySelector(".nav") ||
      document.querySelector(".navbar") ||
      document.querySelector(".top-nav") ||
      document.querySelector(".tabs") ||
      document.querySelector("header");

    if (nav) {
      nav.appendChild(link);
    } else {
      const floatingNav = document.createElement("div");
      floatingNav.id = "ai-studio-floating-nav";
      floatingNav.appendChild(link);
      document.body.appendChild(floatingNav);
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    placeStudioLater();
    addStudioTopNav();
    renderHistory();

    const ask = document.getElementById("studio-ask");
    const clear = document.getElementById("studio-clear");
    const input = document.getElementById("studio-question");
    const output = document.getElementById("studio-output");

    if (ask) ask.addEventListener("click", askStudio);

    if (clear) {
      clear.addEventListener("click", function () {
        if (input) input.value = "";
        if (output) {
          output.innerHTML = `
            <div class="studio-placeholder">
              <div class="studio-placeholder-title">Ready for analysis.</div>
              <p>Ask a recommendation, ranking, comparison, or audio-feature question.</p>
            </div>
          `;
        }
      });
    }

    document.querySelectorAll(".studio-example").forEach(btn => {
      btn.addEventListener("click", function () {
        if (input) input.value = btn.textContent.trim();
      });
    });

    const quick = document.getElementById("music-assistant-toggle");
    if (quick) quick.textContent = "Quick Ask";
  });
})();
</script>
<!-- End AI Music Intelligence Studio -->
"""


def make_system_card_html():
    return """
<!-- System Card -->
<section id="system-card" class="system-card">
  <div class="system-card-inner">
    <div class="system-card-eyebrow">System Card</div>
    <h2>Local Music Intelligence System</h2>
    <p>
      This demo combines an interactive Taylor Swift Spotify report with a local LLM assistant,
      explainable routing, Hybrid RAG recommendation, and deterministic pandas analytics.
    </p>

    <div class="system-card-grid">
      <div class="system-card-item">
        <div class="system-card-label">Model</div>
        <div class="system-card-value">Qwen2.5-3B-Instruct · Local inference · No external API</div>
      </div>

      <div class="system-card-item">
        <div class="system-card-label">Data</div>
        <div class="system-card-value">582 songs · 19 album versions · Spotify audio features + popularity</div>
      </div>

      <div class="system-card-item">
        <div class="system-card-label">Pipelines</div>
        <div class="system-card-value">Auto Router · Hybrid RAG · Pandas Analytics · Evidence-grounded answers</div>
      </div>

      <div class="system-card-item">
        <div class="system-card-label">Reliability</div>
        <div class="system-card-value">Statistics are computed by pandas; recommendations are grounded by retrieved song evidence.</div>
      </div>

      <div class="system-card-item">
        <div class="system-card-label">Limitations</div>
        <div class="system-card-value">Recommendations use Spotify metadata and audio features, not full lyric semantics.</div>
      </div>

      <div class="system-card-item">
        <div class="system-card-label">Feedback Loop</div>
        <div class="system-card-value">User feedback is stored locally for future routing and retrieval quality analysis.</div>
      </div>
    </div>
  </div>
</section>

<style>
#system-card {
  max-width: 1180px;
  margin: 90px auto 120px auto;
  padding: 0 28px;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
}

.system-card-inner {
  border: 1px solid rgba(201,111,138,0.16);
  border-radius: 32px;
  padding: 34px;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,246,250,0.82));
  box-shadow:
    0 24px 70px rgba(92, 67, 96, 0.10),
    inset 0 1px 0 rgba(255,255,255,0.80);
  backdrop-filter: blur(12px);
}

.system-card-eyebrow {
  color: #c96f8a;
  font-size: 0.8rem;
  font-weight: 800;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 10px;
}

#system-card h2 {
  margin: 0 0 12px 0;
  font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
  color: #4b4052;
  font-size: clamp(2rem, 3vw, 3.2rem);
  letter-spacing: -0.04em;
}

#system-card p {
  max-width: 860px;
  color: #6f6474;
  line-height: 1.75;
  margin-bottom: 24px;
}

.system-card-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.system-card-item {
  padding: 16px 18px;
  border-radius: 20px;
  background: rgba(255,255,255,0.68);
  border: 1px solid rgba(201,111,138,0.10);
}

.system-card-label {
  color: #6f91ac;
  font-weight: 800;
  font-size: 0.82rem;
  margin-bottom: 6px;
}

.system-card-value {
  color: #4b4052;
  line-height: 1.55;
  font-size: 0.95rem;
}

@media (max-width: 760px) {
  .system-card-grid {
    grid-template-columns: 1fr;
  }
}
</style>
<!-- End System Card -->
"""


def make_assistant_addons_html():
    return """
<!-- Assistant Feedback + History Addons -->
<style>
.assistant-feedback {
  margin-top: 12px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}

.assistant-feedback-label {
  color: #6f6474;
  font-size: 13px;
  font-weight: 700;
}

.assistant-feedback button,
.assistant-history-item {
  border: 1px solid rgba(201,111,138,0.16);
  background: rgba(255,255,255,0.76);
  border-radius: 999px;
  padding: 7px 11px;
  cursor: pointer;
  color: #4b4052;
  font-size: 13px;
  font-weight: 700;
}

.assistant-feedback-status {
  color: #6f91ac;
  font-size: 13px;
  font-weight: 700;
}

.assistant-history {
  margin-top: 14px;
}


.assistant-history {
  margin-top: 22px !important;
  padding-top: 8px;
  border-top: 1px solid rgba(201,111,138,0.10);
}

.assistant-history .assistant-section-title {
  font-size: 14px !important;
  opacity: 0.78;
}

.assistant-history-list {
  display: grid;
  gap: 7px;
}

.assistant-history-item {
  width: 100%;
  text-align: left;
  border-radius: 14px;
  font-weight: 500;
  line-height: 1.35;
}
</style>

<script>
(function () {
  const HISTORY_KEY = "taylor_music_assistant_history_v1";
  let lastPayload = null;

  function esc(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function getHistory() {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    } catch {
      return [];
    }
  }

  function saveHistory(question) {
    const q = String(question || "").trim();
    if (!q) return;

    let history = getHistory().filter(item => item !== q);
    history.unshift(q);
    history = history.slice(0, 6);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistory();
  }

  function renderHistory() {
    const host = document.getElementById("assistant-history-list");
    if (!host) return;

    const history = getHistory();
    if (history.length === 0) {
      host.innerHTML = '<div class="assistant-muted">No history yet.</div>';
      return;
    }

    host.innerHTML = history.map(q =>
      `<button class="assistant-history-item" data-question="${esc(q)}">${esc(q)}</button>`
    ).join("");

    host.querySelectorAll(".assistant-history-item").forEach(btn => {
      btn.addEventListener("click", function () {
        const question = document.getElementById("assistant-question");
        if (question) question.value = btn.dataset.question || "";
      });
    });
  }

  function ensureHistoryBlock() {
    if (document.getElementById("assistant-history-block")) return;

    const result = document.getElementById("assistant-result");
    const examples = document.querySelector(".assistant-examples");
    const anchor = result || examples;

    if (!anchor) return;

    anchor.insertAdjacentHTML("afterend", `
      <div id="assistant-history-block" class="assistant-history">
        <div class="assistant-section-title">Recent Questions</div>
        <div id="assistant-history-list" class="assistant-history-list"></div>
      </div>
    `);

    renderHistory();
  }

  function appendFeedbackButtons() {
    const result = document.getElementById("assistant-result");
    if (!result || !lastPayload) return;
    if (document.getElementById("assistant-feedback-block")) return;

    result.insertAdjacentHTML("beforeend", `
      <div id="assistant-feedback-block" class="assistant-feedback">
        <span class="assistant-feedback-label">Was this useful?</span>
        <button id="assistant-feedback-up">👍 Helpful</button>
        <button id="assistant-feedback-down">👎 Not Helpful</button>
        <span id="assistant-feedback-status" class="assistant-feedback-status"></span>
      </div>
    `);

    const up = document.getElementById("assistant-feedback-up");
    const down = document.getElementById("assistant-feedback-down");

    up.addEventListener("click", () => sendFeedback("up"));
    down.addEventListener("click", () => sendFeedback("down"));
  }

  async function sendFeedback(value) {
    const status = document.getElementById("assistant-feedback-status");
    if (!lastPayload) return;

    const payload = {
      question: lastPayload.question || "",
      feedback: value,
      intent: lastPayload.intent || "",
      answer: lastPayload.answer || "",
      reliability: lastPayload.reliability || {}
    };

    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });

      if (status) status.textContent = "Saved.";
    } catch (err) {
      if (status) status.textContent = "Failed to save.";
    }
  }

  const originalFetch = window.fetch;
  window.fetch = async function (...args) {
    const response = await originalFetch.apply(this, args);

    try {
      const url = String(args[0] || "");
      if (url.includes("/api/ask")) {
        const clone = response.clone();
        clone.json().then(data => {
          const questionBox = document.getElementById("assistant-question");
          const q = questionBox ? questionBox.value.trim() : "";

          lastPayload = {
            question: q,
            intent: data.intent,
            answer: data.answer,
            reliability: data.reliability
          };

          saveHistory(q);
          setTimeout(appendFeedbackButtons, 120);
        }).catch(() => {});
      }
    } catch {}

    return response;
  };

  document.addEventListener("DOMContentLoaded", function () {
    ensureHistoryBlock();

    const ask = document.getElementById("assistant-ask");
    const question = document.getElementById("assistant-question");

    if (ask && question) {
      ask.addEventListener("click", function () {
        saveHistory(question.value);
      });
    }
  });

  setTimeout(ensureHistoryBlock, 300);
})();
</script>
<!-- End Assistant Feedback + History Addons -->
"""



def make_nav_order_final_fix_html():
    return r"""
<!-- Final Nav + Order Fix -->
<style>
/* 统一顶部导航风格：后加的标签不要做成渐变按钮 */
.demo-native-top-nav .demo-native-extra-link {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;

  background: transparent !important;
  background-image: none !important;
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;

  padding: 0 !important;
  margin: 0 !important;

  color: #5f6d7b !important;
  font-family: inherit !important;
  font-size: inherit !important;
  font-weight: inherit !important;
  letter-spacing: inherit !important;
  line-height: inherit !important;
  text-decoration: none !important;

  transform: none !important;
}

.demo-native-top-nav .demo-native-extra-link:hover {
  color: #4b4052 !important;
  background: transparent !important;
  box-shadow: none !important;
  transform: none !important;
}

/* 隐藏之前单独生成的 AI Studio 按钮 */
#ai-studio-nav-link,
#ai-studio-floating-nav,
#demo-top-nav {
  display: none !important;
}

/* 确保锚点跳转时不要顶到页面最上缘 */
#ai-studio,
#more-analysis,
#system-card,
#overview,
#audio-features,
#era-evolution {
  scroll-margin-top: 96px;
}
</style>

<script>
document.addEventListener("DOMContentLoaded", function () {
  function textOf(el) {
    return (el && el.textContent ? el.textContent : "").trim();
  }

  function forcePageOrder() {
    const ai = document.getElementById("ai-studio");
    const more = document.getElementById("more-analysis");
    const system = document.getElementById("system-card");

    // 最终顺序：AI Studio -> More Analysis -> System Card
    if (system && ai) {
      system.insertAdjacentElement("beforebegin", ai);
    }

    if (ai && more) {
      ai.insertAdjacentElement("afterend", more);
    }

    if (system && more) {
      system.insertAdjacentElement("beforebegin", more);
    }
  }

  function rebuildNativeNav() {
    document.querySelectorAll("#demo-top-nav, #ai-studio-floating-nav").forEach(el => el.remove());

    const clickable = Array.from(document.querySelectorAll("a, button"));
    const overviewLink = clickable.find(el => {
      const t = textOf(el);
      return t.includes("数据概览") || t.includes("Overview");
    });

    const nav = overviewLink ? overviewLink.parentElement : null;
    if (!nav) return;

    nav.classList.add("demo-native-top-nav");

    // 隐藏原来过细的探索性标签
    const hideKeywords = [
      "聚类分析",
      "流行度预测",
      "Era 分类",
      "Era分类",
      "姊妹专辑",
      "Taylor's Version",
      "Taylor’s Version"
    ];

    nav.querySelectorAll("a, button").forEach(el => {
      const t = textOf(el);
      if (hideKeywords.some(k => t.includes(k))) {
        el.style.display = "none";
      }
    });

    // 清掉旧的额外标签，重新按顺序插入
    nav.querySelectorAll(".demo-native-extra-link").forEach(el => el.remove());

    function makeLink(id, label) {
      if (!document.getElementById(id)) return null;

      const a = document.createElement("a");
      a.href = "#" + id;
      a.textContent = label;
      a.className = "demo-native-extra-link";
      return a;
    }

    const links = [
      makeLink("ai-studio", "AI Studio"),
      makeLink("more-analysis", "更多分析"),
      makeLink("system-card", "System Card")
    ].filter(Boolean);

    const visibleLinks = Array.from(nav.querySelectorAll("a, button"))
      .filter(el => el.style.display !== "none");

    let anchor =
      visibleLinks.find(el => textOf(el).includes("风格演变")) ||
      visibleLinks[visibleLinks.length - 1];

    if (!anchor) {
      links.forEach(link => nav.appendChild(link));
      return;
    }

    links.forEach(link => {
      anchor.insertAdjacentElement("afterend", link);
      anchor = link;
    });
  }

  function finalFix() {
    forcePageOrder();
    rebuildNativeNav();
  }

  // 跑两次，覆盖前面其他脚本的二次移动
  setTimeout(finalFix, 0);
  setTimeout(finalFix, 500);
});
</script>
<!-- End Final Nav + Order Fix -->

<!-- Final Nav Spacing Polish -->
<style>
/* 让后加的 AI Studio / 更多分析 / System Card 和前面导航保持素净，但间距更舒服 */
.demo-native-top-nav .demo-native-extra-link {
  margin-left: 34px !important;
  white-space: nowrap !important;
}

/* 如果原导航是 flex/grid，也强制给整体一点横向间距 */
.demo-native-top-nav {
  column-gap: 34px !important;
  row-gap: 12px !important;
}

/* 窄屏时收紧一点 */
@media (max-width: 760px) {
  .demo-native-top-nav .demo-native-extra-link {
    margin-left: 18px !important;
  }

  .demo-native-top-nav {
    column-gap: 18px !important;
  }
}
</style>
<!-- End Final Nav Spacing Polish -->

"""


@app.get("/", response_class=HTMLResponse)
def report_root():
    if not REPORT_PATH.exists():
        return HTMLResponse("<h1>Report file not found</h1>", status_code=404)

    html = REPORT_PATH.read_text(encoding="utf-8")

    # 移除之前试验过的 iframe 版组件或本服务组件，避免重复注入
    html = re.sub(
        r"<!-- Floating Music Assistant -->.*?<!-- End Floating Music Assistant -->",
        "",
        html,
        flags=re.S,
    )
    html = re.sub(
        r"<!-- Native Music Assistant -->.*?<!-- End Native Music Assistant -->",
        "",
        html,
        flags=re.S,
    )

    html = html.replace("</body>", make_demo_focus_layout_html() + make_ai_studio_html() + make_system_card_html() + make_widget_html() + make_assistant_addons_html() + make_nav_order_final_fix_html() + "\n</body>")
    return HTMLResponse(html)


@app.get("/taylor_swift_interactive_report.html", response_class=HTMLResponse)
def report_html():
    return report_root()



@app.post("/api/feedback")
def feedback_api(payload: FeedbackRequest):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": payload.question,
        "feedback": payload.feedback,
        "intent": payload.intent,
        "answer": payload.answer[:1000],
        "reliability": payload.reliability,
    }

    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return JSONResponse({"ok": True})


@app.post("/api/ask")
def ask_api(payload: AskRequest):
    question = payload.question.strip()

    if not question:
        return JSONResponse({
            "intent": "",
            "answer": "请输入问题。",
            "rows": [],
        })

    result = answer_auto(question, top_k=payload.top_k)
    intent = result["intent"]

    if intent == "analytics":
        rows = dataframe_to_records(result.get("table"))
    else:
        rows = []
        for item in result.get("retrieved", []):
            m = item.get("metadata", {})
            rows.append({
                "name": m.get("name", ""),
                "album": m.get("album", ""),
                "version": m.get("version_type", ""),
                "energy": round(float(m.get("energy", 0)), 3),
                "valence": round(float(m.get("valence", 0)), 3),
                "acousticness": round(float(m.get("acousticness", 0)), 3),
            })

    return JSONResponse({
        "intent": intent,
        "answer": result["answer"],
        "rows": rows,
        "route_reason": result.get("route_reason", ""),
        "pipeline": result.get("pipeline", ""),
        "signals": result.get("signals", {}),
        "reliability": result.get("reliability", {}),
        "trace": result.get("trace", []),
    })
