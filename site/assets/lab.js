/* The phone menu: the header's links on a graphite overlay. Without JavaScript they simply wrap. */
(function () {
  "use strict";
  const nav = document.querySelector(".lab-nav");
  if (!nav) return;
  const open = nav.querySelector(".lab-nav__open"), close = nav.querySelector(".lab-nav__close");
  const set = (on) => {
    nav.classList.toggle("is-open", on);
    open.setAttribute("aria-expanded", String(on));
    document.body.style.overflow = on ? "hidden" : "";
    (on ? close : open).focus();
  };
  open.addEventListener("click", () => set(true));
  close.addEventListener("click", () => set(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && nav.classList.contains("is-open")) set(false); });
})();
