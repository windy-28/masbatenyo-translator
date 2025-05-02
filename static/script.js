document.addEventListener("DOMContentLoaded", function () {
    const privacyCheckbox = document.getElementById("privacy");
    const startBtn = document.getElementById("startBtn");

    privacyCheckbox.addEventListener("change", function () {
        if (this.checked) {
            startBtn.classList.add("active");
            startBtn.removeAttribute("disabled");
        } else {
            startBtn.classList.remove("active");
            startBtn.setAttribute("disabled", "true");
        }
    });
});
