/** Backend API (FastAPI). Start: uvicorn api_server:app --reload --port 8000 */
const API_BASE_URL = 'http://localhost:8000';

// ——— Nav scroll ———
const navEl = document.getElementById('nav');
window.addEventListener('scroll', () => {
  navEl.classList.toggle('scrolled', window.scrollY > 40);
});

// ——— Mobile menu ———
const menuToggle = document.getElementById('menuToggle');
const navPanel = document.getElementById('navPanel');

function closeMenu() {
  document.body.classList.remove('menu-open');
  if (menuToggle) {
    menuToggle.setAttribute('aria-expanded', 'false');
    menuToggle.setAttribute('aria-label', 'Open menu');
  }
}

function openMenu() {
  document.body.classList.add('menu-open');
  if (menuToggle) {
    menuToggle.setAttribute('aria-expanded', 'true');
    menuToggle.setAttribute('aria-label', 'Close menu');
  }
}

if (menuToggle) {
  menuToggle.addEventListener('click', () => {
    if (document.body.classList.contains('menu-open')) closeMenu();
    else openMenu();
  });
}

document.querySelectorAll('.nav-links a[data-nav]').forEach((a) => {
  a.addEventListener('click', () => closeMenu());
});

// ——— Page switching ———
function showDemo() {
  document.getElementById('home').hidden = true;
  const demo = document.getElementById('demo-page');
  demo.hidden = false;
  window.scrollTo(0, 0);
  closeMenu();
}

function showHome() {
  document.getElementById('demo-page').hidden = true;
  document.getElementById('home').hidden = false;
  window.scrollTo(0, 0);
}

// ——— Toast ———
function showToast(message) {
  const t = document.getElementById('toast');
  t.textContent = message;
  t.classList.add('show');
  clearTimeout(showToast._tid);
  showToast._tid = setTimeout(() => t.classList.remove('show'), 3200);
}

// ——— Scroll spy (nav highlights) ———
const sections = ['hero', 'problem', 'solution', 'pipeline', 'tech', 'team'];
const navLinks = document.querySelectorAll('.nav-links a[data-nav]');

function updateActiveNav() {
  const demoOpen = !document.getElementById('demo-page').hidden;
  if (demoOpen) {
    navLinks.forEach((a) => a.classList.remove('active'));
    return;
  }
  const y = window.scrollY + 120;
  let current = 'hero';
  for (const id of sections) {
    const el = document.getElementById(id);
    if (!el) continue;
    if (el.offsetTop <= y) current = id;
  }
  navLinks.forEach((a) => {
    const href = a.getAttribute('href');
    a.classList.toggle('active', href === `#${current}`);
  });
}
window.addEventListener('scroll', updateActiveNav, { passive: true });
window.addEventListener('resize', updateActiveNav);
updateActiveNav();

// ——— Reveal on scroll ———
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (!reduceMotion) {
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) e.target.classList.add('is-visible');
      });
    },
    { rootMargin: '0px 0px -8% 0px', threshold: 0.08 }
  );
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el));
} else {
  document.querySelectorAll('.reveal').forEach((el) => el.classList.add('is-visible'));
}

// ——— Particle canvas ———
const canvas = document.getElementById('particles');
const ctx = canvas.getContext('2d');
let particles = [];
let rafId = null;

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function initParticles() {
  particles = [];
  const n = reduceMotion ? 0 : 55;
  for (let i = 0; i < n; i++) {
    particles.push({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5,
      o: Math.random() * 0.5 + 0.1,
    });
  }
}

function drawParticles() {
  if (reduceMotion || !particles.length) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach((p) => {
    p.x += p.vx;
    p.y += p.vy;
    if (p.x < 0) p.x = canvas.width;
    if (p.x > canvas.width) p.x = 0;
    if (p.y < 0) p.y = canvas.height;
    if (p.y > canvas.height) p.y = 0;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(26,83,70,${p.o})`;
    ctx.fill();
  });
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const dx = particles[i].x - particles[j].x;
      const dy = particles[i].y - particles[j].y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < 120) {
        ctx.beginPath();
        ctx.strokeStyle = `rgba(26,83,70,${0.08 * (1 - d / 120)})`;
        ctx.lineWidth = 0.5;
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(particles[j].x, particles[j].y);
        ctx.stroke();
      }
    }
  }
  rafId = requestAnimationFrame(drawParticles);
}

if (canvas) {
  resizeCanvas();
  window.addEventListener('resize', () => {
    resizeCanvas();
    initParticles();
  });
  initParticles();
  if (!reduceMotion) drawParticles();
}

// ——— Pipeline ———
const pipelineSteps = [
  { num: '01', icon: '🖼️', title: 'Sketch Input', model: 'OpenCLIP ViT-H-14', desc: 'Vision model extracts semantic feature tags from the uploaded sketch, converting visual concepts into a 512-dimensional embedding vector.', output: '512-d image vector', color: '#1a5346' },
  { num: '02', icon: '💬', title: 'NLP Encoding', model: 'mxbai-embed-large-v1', desc: 'NLP model encodes the plain-English description into a 1024-dimensional text embedding. Ranked #1 on the MTEB benchmark.', output: '1024-d text vector', color: '#4a5d7a' },
  { num: '03', icon: '⚙️', title: 'Vector Fusion', model: 'PyTorch Fusion', desc: 'Both vectors are combined using weighted concatenation and L2 normalization into a single unified query vector representing the full semantic meaning.', output: 'Unified query vector', color: '#6b5b7a' },
  { num: '04', icon: '🔍', title: 'Semantic Retrieval', model: 'FAISS + Qdrant', desc: 'The unified vector queries 50K+ indexed patents from Google Patents, USPTO, and WIPO/EPO. FAISS retrieves the top 100 candidates.', output: 'Top 100 candidates', color: '#2d7a68' },
  { num: '05', icon: '🎯', title: 'Re-ranking', model: 'ms-marco-MiniLM', desc: 'A cross-encoder re-ranking model scores the top 100 candidates and selects the 10 most semantically similar patents.', output: 'Top 10 patents', color: '#1d6b52' },
  { num: '06', icon: '📋', title: 'Novelty Report', model: 'Gemini 2.5 Flash', desc: 'Gemini generates plain-English summaries, novelty scores, claim overlap analysis, differentiating features, and IPC classification per patent.', output: 'Shareable report', color: '#9a5f3c' },
];

let activeStep = 0;

function renderPipeline() {
  document.getElementById('pipeSteps').innerHTML = pipelineSteps
    .map(
      (s, i) => `
    <button type="button" class="ps ${i === activeStep ? 'active' : ''}" onclick="setStep(${i})">
      <span class="ps-num">${s.num}</span>
      <span>${s.icon} ${s.title}</span>
    </button>
  `
    )
    .join('');

  const s = pipelineSteps[activeStep];
  document.getElementById('pipeDetail').innerHTML = `
    <div class="pd-head">
      <span class="pd-icon">${s.icon}</span>
      <div>
        <div class="pd-title">${s.title}</div>
        <div class="pd-model"><code>${s.model}</code></div>
      </div>
    </div>
    <p class="pd-desc">${s.desc}</p>
    <div class="pd-out">
      <span class="out-l">Output</span>
      <span class="out-v" style="color:${s.color}">→ ${s.output}</span>
    </div>
    <div class="pd-nav">
      <button type="button" class="nb" onclick="setStep(${activeStep - 1})" ${activeStep === 0 ? 'disabled' : ''}>← Prev</button>
      <div class="ndots">
        ${pipelineSteps
          .map(
            (_, i) => `
          <button type="button" class="nd ${i === activeStep ? 'active' : ''}" onclick="setStep(${i})"
            style="${i === activeStep ? `background:${s.color}` : ''}" aria-label="Stage ${i + 1}"></button>
        `
          )
          .join('')}
      </div>
      <button type="button" class="nb" onclick="setStep(${activeStep + 1})" ${activeStep === pipelineSteps.length - 1 ? 'disabled' : ''}>Next →</button>
    </div>
  `;
}

function setStep(i) {
  if (i < 0 || i >= pipelineSteps.length) return;
  activeStep = i;
  renderPipeline();
}

renderPipeline();

const pipelineWrap = document.getElementById('pipelineWrap');
if (pipelineWrap) {
  pipelineWrap.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      setStep(activeStep + 1);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setStep(activeStep - 1);
    }
  });
}

// ——— Demo: file upload ———
function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = document.getElementById('previewImg');
    img.src = ev.target.result;
    img.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

function dragOver(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.add('drag-over');
}
function dragLeave() {
  document.getElementById('dropZone').classList.remove('drag-over');
}
function dropFile(e) {
  e.preventDefault();
  dragLeave();
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = document.getElementById('previewImg');
      img.src = ev.target.result;
      img.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }
}

// attach drag handlers (inline ondrop can stay; ensure dragover works)
const dropZone = document.getElementById('dropZone');
if (dropZone) {
  dropZone.addEventListener('dragover', dragOver);
  dropZone.addEventListener('dragleave', dragLeave);
  dropZone.addEventListener('drop', dropFile);
}

const mockResults = [
  {
    id: 'US10,891,344',
    title: 'Automated patent prior art identification using neural embeddings',
    db: 'USPTO',
    ipc: 'G06F 16/903',
    year: '2021',
    novelty: 74,
    abstract:
      'A system for identifying prior art by encoding patent claims into semantic embedding space and performing similarity-based retrieval. Uses transformer-based encoders for text and image inputs.',
    diff: 'Does not combine sketch input with text encoding. No vector fusion or L2 normalization. No plain-English report generation.',
  },
  {
    id: 'WO2022/087654',
    title: 'Multi-modal patent search with image-text fusion for novelty assessment',
    db: 'WIPO',
    ipc: 'G06N 3/04',
    year: '2022',
    novelty: 61,
    abstract:
      'A patent search platform combining image and text inputs via late fusion of embedding vectors. Retrieves semantically similar patents and generates novelty reports using a language model.',
    diff: 'Late fusion without L2 normalization. No dedicated re-ranking stage. Targets corporate users only, not individual inventors or MSMEs.',
  },
  {
    id: 'US9,547,881',
    title: 'Automated IPC classification of patent documents using machine learning',
    db: 'USPTO',
    ipc: 'G06F 40/30',
    year: '2017',
    novelty: 89,
    abstract: 'A method for automatically classifying patent documents into IPC codes using supervised machine learning models trained on a labeled patent corpus.',
    diff: 'Classification only — no retrieval, no novelty scoring, no sketch input. Entirely different scope from AviShkar.',
  },
  {
    id: 'EP3,891,643',
    title: 'CLIP-based visual feature extraction for design patent search',
    db: 'EPO',
    ipc: 'G06V 10/764',
    year: '2023',
    novelty: 78,
    abstract:
      'Application of CLIP models to extract visual features from design patent images for similarity-based search, enabling cross-modal retrieval between text queries and patent figures.',
    diff: 'Visual search only. No text fusion, no IPC mapping, no novelty report generation. Targets design patents, not utility patents.',
  },
];

const stages = [
  'Extracting visual features with OpenCLIP ViT-H-14…',
  'Encoding text with mxbai-embed-large-v1…',
  'Fusing vectors — weighted concat + L2 norm…',
  'Querying 50K+ patents via FAISS…',
  'Re-ranking top 100 with ms-marco-MiniLM…',
  'Generating novelty report with Gemini 2.5 Flash…',
];

let activeResult = 0;
let searchAbort = null;
let mockSearchInterval = null;

function setProg(pct) {
  const fill = document.getElementById('progFill');
  const track = document.getElementById('progTrack');
  if (fill) fill.style.width = `${pct}%`;
  if (track) track.setAttribute('aria-valuenow', String(Math.round(pct)));
}

async function runSearch() {
  const text = document.getElementById('ideaText').value.trim();
  const preview = document.getElementById('previewImg');
  const hasFile = preview && preview.style.display !== 'none' && preview.src;

  if (!text && !hasFile) {
    showToast('Add a short description of your idea or upload a sketch.');
    document.getElementById('ideaText').focus();
    return;
  }

  document.getElementById('emptyState').hidden = true;
  document.getElementById('resultsBox').hidden = true;
  document.getElementById('loadingBox').hidden = false;
  document.getElementById('searchBtn').disabled = true;
  setProg(0);

  if (searchAbort) searchAbort.abort();
  searchAbort = new AbortController();

  if (API_BASE_URL) {
    try {
      const fd = new FormData();
      if (text) fd.append('text', text);
      const fileInput = document.getElementById('fileInput');
      if (fileInput.files[0]) fd.append('file', fileInput.files[0]);

      setProg(15);
      const res = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        body: fd,
        signal: searchAbort.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const list = Array.isArray(data.results) ? data.results : data;
      if (!list.length) throw new Error(data.error || 'No results');
      window.__lastResults = list;
      setProg(100);
      document.getElementById('loadingBox').hidden = true;
      document.getElementById('resultsBox').hidden = false;
      document.getElementById('searchBtn').disabled = false;
      activeResult = 0;
      renderResultsFrom(window.__lastResults);
      return;
    } catch (err) {
      if (err.name === 'AbortError') return;
      showToast(err.message || 'Backend error — showing mock data.');
    }
  }

  if (mockSearchInterval) clearInterval(mockSearchInterval);

  document.getElementById('stageList').innerHTML = stages
    .map(
      (s, i) => `
    <div class="stage-item" id="stage${i}">
      <span class="sdot"></span>${s}
    </div>
  `
    )
    .join('');

  let current = 0;
  mockSearchInterval = setInterval(() => {
    if (current > 0) document.getElementById(`stage${current - 1}`).className = 'stage-item done';
    if (current < stages.length) {
      document.getElementById(`stage${current}`).className = 'stage-item running';
      setProg(((current + 1) / stages.length) * 100);
      current++;
    } else {
      clearInterval(mockSearchInterval);
      mockSearchInterval = null;
      document.getElementById(`stage${stages.length - 1}`).className = 'stage-item done';
      setTimeout(() => {
        window.__lastResults = mockResults;
        showResults();
      }, 400);
    }
  }, 700);
}

function showResults() {
  document.getElementById('loadingBox').hidden = true;
  document.getElementById('resultsBox').hidden = false;
  document.getElementById('searchBtn').disabled = false;
  activeResult = 0;
  renderResultsFrom(window.__lastResults || mockResults);
}

function renderResultsFrom(list) {
  const results = list || mockResults;
  document.getElementById('resultTabs').innerHTML = results
    .map(
      (r, i) => `
    <button type="button" role="tab" class="rtab ${i === activeResult ? 'active' : ''}" onclick="setResult(${i})" aria-selected="${i === activeResult}">${r.id}</button>
  `
    )
    .join('');

  const r = results[activeResult];
  const color = r.novelty >= 80 ? '#1d6b52' : r.novelty >= 65 ? '#b8892a' : '#b54747';
  const label = r.novelty >= 80 ? 'High Novelty' : r.novelty >= 65 ? 'Moderate Overlap' : 'Low Novelty';

  document.getElementById('resultCard').innerHTML = `
    <div class="result-card">
      <div class="rc-id">${r.id}</div>
      <div class="rc-title">${r.title}</div>
      <div class="rc-chips">
        <span class="chip db">${r.db}</span>
        <span class="chip">IPC: ${r.ipc}</span>
        <span class="chip">${r.year}</span>
      </div>
      <div class="nov-label">Novelty Score</div>
      <div class="nov-track"><div class="nov-fill" style="width:${r.novelty}%;background:${color}"></div></div>
      <div class="nov-score" style="color:${color}">${r.novelty}% — ${label}</div>
      <p class="rc-abstract">${r.abstract}</p>
      <div class="diff-label">How your idea differs</div>
      <div class="diff-box">${r.diff}</div>
    </div>
  `;
}

function renderResults() {
  renderResultsFrom(window.__lastResults || mockResults);
}

function setResult(i) {
  const list = window.__lastResults || mockResults;
  if (i < 0 || i >= list.length) return;
  activeResult = i;
  renderResults();
}
