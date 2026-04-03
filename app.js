const state = {
  rows: [],
  filtered: [],
  page: 1,
  perPage: 40,
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
const quizBtn = document.getElementById("quizBtn");
const quizPanel = document.getElementById("quizPanel");
const quizCards = document.getElementById("quizCards");

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

function renderQuiz() {
  const src = state.filtered.length > 0 ? state.filtered : state.rows;
  const picked = [...src].sort(() => Math.random() - 0.5).slice(0, 5);

  quizCards.innerHTML = picked
    .map(
      (r, i) => `
      <article class="card" style="animation-delay:${i * 40}ms">
        <div class="word">${escapeHtml(r.word)}</div>
        <div>${escapeHtml(r.ipa)} / ${escapeHtml(r.pos)}</div>
        <div>意味: ${escapeHtml(r.meaning_ja)}</div>
        <div style="margin-top:6px;color:#415055;">${escapeHtml(r.example_en)}</div>
      </article>
    `
    )
    .join("");

  quizPanel.hidden = false;
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

quizBtn.addEventListener("click", renderQuiz);

init();
