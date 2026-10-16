const fileInput = document.querySelector("#fileInput");
const dropzone = document.querySelector("#dropzone");
const emptyState = document.querySelector("#emptyState");
const preview = document.querySelector("#preview");
const changeImage = document.querySelector("#changeImage");
const diagnoseButton = document.querySelector("#diagnoseButton");
const confidence = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidenceValue");
const idleResult = document.querySelector("#idleResult");
const errorResult = document.querySelector("#errorResult");
const diagnosisList = document.querySelector("#diagnosisList");
const resultCount = document.querySelector("#resultCount");
const disclaimer = document.querySelector("#disclaimer");

let selectedFile = null;
let previewUrl = null;

function chooseFile() {
  fileInput.click();
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Vui lòng chọn một tệp ảnh hợp lệ.");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    showError("Ảnh vượt quá giới hạn 12 MB.");
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  emptyState.hidden = true;
  changeImage.hidden = false;
  diagnoseButton.disabled = false;
  resetResults();
}

function resetResults() {
  idleResult.hidden = false;
  errorResult.hidden = true;
  diagnosisList.hidden = true;
  disclaimer.hidden = true;
  resultCount.textContent = "—";
}

function showError(message) {
  idleResult.hidden = true;
  diagnosisList.hidden = true;
  disclaimer.hidden = true;
  errorResult.textContent = message;
  errorResult.hidden = false;
  resultCount.textContent = "!";
}

function renderResults(data) {
  idleResult.hidden = true;
  errorResult.hidden = true;
  diagnosisList.innerHTML = "";
  resultCount.textContent = String(data.count).padStart(2, "0");
  preview.src = data.annotated_image;

  if (!data.diagnoses.length) {
    diagnosisList.innerHTML = `<div class="no-damage">Không phát hiện hư hỏng vượt ngưỡng hiện tại.<br>Hãy hạ ngưỡng nếu cần kiểm tra kỹ hơn.</div>`;
  } else {
    for (const item of data.diagnoses) {
      const card = document.createElement("article");
      card.className = "diagnosis-card";
      card.innerHTML = `
        <div class="diagnosis-top">
          <div>
            <div class="diagnosis-index">FINDING ${String(item.id).padStart(2, "0")}</div>
            <h3>${item.damage_vi}</h3>
            <div class="panel-name">${item.panel_vi}</div>
          </div>
          <div class="confidence-pill">${Math.round(item.confidence * 100)}%</div>
        </div>
        <div class="diagnosis-meta">
          <span>Mức độ: ${item.severity}</span>
          <span>Panel match: ${Math.round(item.panel_match * 100)}%</span>
        </div>
        <p>${item.action}</p>`;
      diagnosisList.appendChild(card);
    }
  }
  diagnosisList.hidden = false;
  disclaimer.textContent = data.disclaimer;
  disclaimer.hidden = false;
}

async function diagnose() {
  if (!selectedFile) return;
  diagnoseButton.disabled = true;
  diagnoseButton.classList.add("loading");
  errorResult.hidden = true;
  const form = new FormData();
  form.append("image", selectedFile);
  form.append("confidence", String(Number(confidence.value) / 100));

  try {
    const response = await fetch("/api/diagnose", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Không thể phân tích ảnh.");
    renderResults(data);
  } catch (error) {
    showError(error.message || "Đã xảy ra lỗi khi phân tích.");
  } finally {
    diagnoseButton.disabled = false;
    diagnoseButton.classList.remove("loading");
  }
}

dropzone.addEventListener("click", (event) => {
  if (event.target !== changeImage) chooseFile();
});
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") chooseFile();
});
changeImage.addEventListener("click", (event) => {
  event.stopPropagation();
  chooseFile();
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
diagnoseButton.addEventListener("click", diagnose);
confidence.addEventListener("input", () => {
  confidenceValue.textContent = `${confidence.value}%`;
});

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
