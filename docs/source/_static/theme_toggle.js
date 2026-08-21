(function () {
  var STORAGE_KEY = "dc-docs-theme";
  var MODES = ["auto", "dark", "light"];

  function getStoredMode() {
    var mode = window.localStorage.getItem(STORAGE_KEY);
    return MODES.indexOf(mode) === -1 ? "auto" : mode;
  }

  function setTheme(mode) {
    if (mode === "auto") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", mode);
    }
    window.localStorage.setItem(STORAGE_KEY, mode);
    return mode;
  }

  function updateButtonLabel(button, mode) {
    button.textContent = "Theme: " + mode;
    button.setAttribute("aria-label", "Theme mode is " + mode + ". Click to cycle.");
    button.setAttribute("title", "Theme mode: " + mode);
  }

  function nextMode(mode) {
    var idx = MODES.indexOf(mode);
    return MODES[(idx + 1) % MODES.length];
  }

  function initThemeToggle() {
    var mode = setTheme(getStoredMode());
    var button = document.createElement("button");
    button.className = "dc-theme-toggle";
    button.type = "button";
    updateButtonLabel(button, mode);

    button.addEventListener("click", function () {
      mode = setTheme(nextMode(mode));
      updateButtonLabel(button, mode);
    });

    document.body.appendChild(button);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initThemeToggle);
  } else {
    initThemeToggle();
  }
})();
