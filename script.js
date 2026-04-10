const actionBtn = document.getElementById("actionBtn");
const status = document.getElementById("status");

if (actionBtn && status) {
  actionBtn.addEventListener("click", () => {
    status.textContent = "Кнопка нажата. Логика script.js работает.";
  });
}
