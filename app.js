const state = {
  rows: [],
  filtered: [],
  page: 1,
  perPage: 40,
  quizPool: [],
  quizIndex: 0,
  wrongRows: [],
  quizRound: 1,
};

const tbody = document.getElementById("tbody");
const search = document.getElementById("search");
const posFilter = document.getElementById("posFilter");
const sortBy = document.getElementById("sortBy");
const pageInfo = document.getElementById("pageInfo");
const totalCount = document.getElementById("totalCount");
const visibleCount = document.getElementById("visibleCount");
const prevPage = document.getElementById("prevPage");
const nextPage = document.getElementById("nextPage");
const quizStartBtn = document.getElementById("quizStartBtn");
const quizPanel = document.getElementById("quizPanel");
const quizRound = document.getElementById("quizRound");
const quizProgress = document.getElementById("quizProgress");
const quizWrongCount = document.getElementById("quizWrongCount");
const quizWord = document.getElementById("quizWord");
const quizMeta = document.getElementById("quizMeta");
const showAnswerBtn = document.getElementById("showAnswerBtn");
const markCorrectBtn = document.getElementById("markCorrectBtn");
const markWrongBtn = document.getElementById("markWrongBtn");
const quizAnswer = document.getElementById("quizAnswer");
const answerMeaning = document.getElementById("answerMeaning");
const answerIpa = document.getElementById("answerIpa");
const answerExample = document.getElementById("answerExample");
const nextRoundBtn = document.getElementById("nextRoundBtn");
const restartQuizBtn = document.getElementById("restartQuizBtn");

function applyFilters() {
  const q = search.value.trim().toLowerCase();
  const pos = posFilter.value;

  state.filtered = state.rows.filter((r) => {
    const hitText = `${r.word} ${r.ipa} ${r.pos} ${r.meaning_ja} ${r.example_en}`.toLowerCase();
    const okQuery = q === "" || hitText.includes(q);
    const okPos = pos === "all" || r.pos === pos;
    return okQuery && okPos;
  });

  if (sortBy.value === "word") {
    state.filtered.sort((a, b) => a.word.localeCompare(b.word));
  } else {
    state.filtered.sort((a, b) => a.id - b.id);
  }

  const maxPage = Math.max(1, Math.ceil(state.filtered.length / state.perPage));
  state.page = Math.min(state.page, maxPage);
  renderTable();
}

function renderTable() {
  const start = (state.page - 1) * state.perPage;
  const pageRows = state.filtered.slice(start, start + state.perPage);

  tbody.innerHTML = pageRows
    .map(
      (r) => `
      <tr>
        <td>${r.id}</td>
        <td>${escapeHtml(r.word)}</td>
        <td>${escapeHtml(r.ipa)}</td>
        <td>${escapeHtml(r.pos)}</td>
        <td>${escapeHtml(r.meaning_ja)}</td>
        <td>${escapeHtml(r.example_en)}</td>
      </tr>
    `
    )
    .join("");

  const maxPage = Math.max(1, Math.ceil(state.filtered.length / state.perPage));
  pageInfo.textContent = `${state.page} / ${maxPage}`;
  totalCount.textContent = `${state.rows.length} words`;
  visibleCount.textContent = `${state.filtered.length} shown`;
}

function shuffleRows(rows) {
  const arr = [...rows];
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
  }
  return arr;
}

function startQuizAll() {
  state.quizRound = 1;
  state.wrongRows = [];
  state.quizPool = shuffleRows(state.rows);
  state.quizIndex = 0;
  quizPanel.hidden = false;
  nextRoundBtn.hidden = true;
  restartQuizBtn.hidden = true;
  renderQuizQuestion();
}

function renderQuizQuestion() {
  const total = state.quizPool.length;
  const current = state.quizPool[state.quizIndex];

  if (!current) {
    quizWord.textContent = "Round complete";
    quizMeta.textContent = `Wrong answers: ${state.wrongRows.length}`;
    quizProgress.textContent = `${total} / ${total}`;
    quizWrongCount.textContent = `Wrong: ${state.wrongRows.length}`;
    quizAnswer.hidden = true;
    showAnswerBtn.disabled = true;
    markCorrectBtn.disabled = true;
    markWrongBtn.disabled = true;
    nextRoundBtn.hidden = state.wrongRows.length === 0;
    restartQuizBtn.hidden = false;
    return;
  }

  quizRound.textContent = `Round ${state.quizRound}`;
  quizProgress.textContent = `${state.quizIndex + 1} / ${total}`;
  quizWrongCount.textContent = `Wrong: ${state.wrongRows.length}`;
  quizWord.textContent = current.word;
  quizMeta.textContent = `${current.pos}`;

  answerMeaning.textContent = current.meaning_ja;
  answerIpa.textContent = current.ipa;
  answerExample.textContent = current.example_en;

  quizAnswer.hidden = true;
  showAnswerBtn.disabled = false;
  markCorrectBtn.disabled = true;
  markWrongBtn.disabled = true;
}

function moveNext(correct) {
  const current = state.quizPool[state.quizIndex];
  if (!current) {
    return;
  }

  if (!correct) {
    state.wrongRows.push(current);
  }

  state.quizIndex += 1;
  renderQuizQuestion();
}

function startWrongRound() {
  state.quizRound += 1;
  state.quizPool = shuffleRows(state.wrongRows);
  state.wrongRows = [];
  state.quizIndex = 0;
  nextRoundBtn.hidden = true;
  restartQuizBtn.hidden = true;
  renderQuizQuestion();
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function init() {
  const res = await fetch("./data/engineering_vocab_2000.json");
  state.rows = await res.json();
  state.filtered = [...state.rows];
  renderTable();
}

search.addEventListener("input", () => {
  state.page = 1;
  applyFilters();
});
posFilter.addEventListener("change", () => {
  state.page = 1;
  applyFilters();
});
sortBy.addEventListener("change", applyFilters);

prevPage.addEventListener("click", () => {
  state.page = Math.max(1, state.page - 1);
  renderTable();
});

nextPage.addEventListener("click", () => {
  const maxPage = Math.max(1, Math.ceil(state.filtered.length / state.perPage));
  state.page = Math.min(maxPage, state.page + 1);
  renderTable();
});

quizStartBtn.addEventListener("click", startQuizAll);

showAnswerBtn.addEventListener("click", () => {
  quizAnswer.hidden = false;
  markCorrectBtn.disabled = false;
  markWrongBtn.disabled = false;
});

markCorrectBtn.addEventListener("click", () => moveNext(true));
markWrongBtn.addEventListener("click", () => moveNext(false));
nextRoundBtn.addEventListener("click", startWrongRound);
restartQuizBtn.addEventListener("click", startQuizAll);

init();
