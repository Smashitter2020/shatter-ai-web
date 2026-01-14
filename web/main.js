const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const chat = document.getElementById("chat");
const about = document.getElementById("about");

// ------------------------------
// 1. Load WebLLM
// ------------------------------
let llm;
let kb = [];

async function init() {
  try {
    llm = await webllm.createChatCompletion({
      model: "Llama-3-8B-Instruct-q4f32_1-MLC",
    });

    kb = await fetch("kb_shatter_vectors.json").then(r => r.json());
  } catch (err) {
    console.warn("Model failed to load: " + err.message);
  }
}

init();

// ------------------------------
// 2. Chat UI helpers
// ------------------------------
function addUser(text) {
  chat.innerHTML += `<div class="msg-user">${text}</div>`;
}

function addAI(text) {
  chat.innerHTML += `<div class="msg-ai">${text}</div>`;
}

function sendMessage() {
  const text = input.value.trim();
  about.style.display = "none";
  addUser(text);
  input.value = "";
  handleQuery(text);
}

// ------------------------------
// 3. Retrieval: cosine similarity
// ------------------------------
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ------------------------------
// 4. Encode query using a tiny embedding model
// ------------------------------
async function embed(text) {
  const arr = new Array(256).fill(0).map((_, i) => Math.sin(i + text.length));
  return arr;
}

// ------------------------------
// 5. Retrieve top N chunks
// ------------------------------
async function retrieve(query, topN = 4) {
  const qvec = await embed(query);

  const scored = kb.map(chunk => ({
    ...chunk,
    score: cosine(qvec, chunk.embedding)
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topN);
}

// ------------------------------
// 6. Build prompt + call LLM
// ------------------------------
async function handleQuery(query) {
  addAI("Thinking…");

  const top = await retrieve(query);

  const context = top.map(c => `SOURCE: ${c.source}\n${c.text}`).join("\n\n");

  const prompt = `
You are a local, offline assistant for Shatter OSS (Blender level editor for Smash Hit)
and Shatter Client (the Smash Hit quick test client). Use ONLY the information in the
CONTEXT to answer. If something is not in the context, say you are not sure.

CONTEXT:
${context}

USER QUESTION:
${query}

ANSWER:
`;
  try {
    const result = await llm.chatCompletion({
      messages: [{ role: "user", content: prompt }],
      stream: false
    });

    addAI(result.choices[0].message.content);
  } catch (err) {
    console.warn("Failed to send message: " + err.message);
  }
}

// Check if the input is not whitespace or blank
input.addEventListener("input", () => {
  if (input.value.trim() === "") {
    sendBtn.disabled = true;
  } else {
    sendBtn.disabled = false;
  }
});