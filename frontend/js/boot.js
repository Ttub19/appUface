// Loads each screen's markup from pages/*.html into its placeholder, then starts the app.
// The app stays a single page: every screen is in the DOM after boot and navigateTo()
// only toggles visibility, so state survives moving between screens.
//
// fetch() does not work from file://, so open the site through a local server
// (npm start) or GitHub Pages.

const PAGES_VERSION = "20261114";

async function loadPage(placeholder) {
    const src = placeholder.dataset.include;
    const res = await fetch(`${src}?v=${PAGES_VERSION}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${src}: HTTP ${res.status}`);
    const html = await res.text();
    const fragment = document.createRange().createContextualFragment(html);
    placeholder.replaceWith(fragment);
}

function showBootError(err) {
    console.error("[boot]", err);
    const box = document.createElement("div");
    box.className = "boot-error";
    box.setAttribute("role", "alert");
    box.innerHTML =
        "<h1>โหลดหน้าเว็บไม่ครบ</h1>" +
        "<p>ลองรีเฟรชอีกครั้ง ถ้าเปิดไฟล์จากเครื่องโดยตรง ให้รันผ่าน <code>npm start</code> แทน</p>" +
        '<button type="button" class="btn btn-primary">รีเฟรช</button>';
    box.querySelector("button").addEventListener("click", () => location.reload());
    document.body.prepend(box);
}

async function boot() {
    const placeholders = [...document.querySelectorAll("[data-include]")];
    try {
        await Promise.all(placeholders.map(loadPage));
    } catch (err) {
        showBootError(err);
        return;
    }
    await import("./app.js");
}

boot();
