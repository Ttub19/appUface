import { faceData } from "../data.js";

export function renderGuide() {
    const grid = document.getElementById("guide-grid");
    if (grid.dataset.rendered) return;
    grid.dataset.rendered = "1";
    faceData.forEach((shape) => {
        const card = document.createElement("div");
        card.className = "bg-white rounded-3xl border border-slate-200 shadow-sm p-6 space-y-3";
        card.innerHTML = `
            <div class="flex items-center justify-between">
                <h4 class="font-bold text-slate-800 font-display">${shape.name}</h4>
                <span class="w-3 h-3 rounded-full bg-[#4274D9]"></span>
            </div>
            <p class="text-sm text-slate-500 leading-relaxed">${shape.desc}</p>
            <button onclick="toggleGuideDetail('${shape.id}')" class="text-xs text-[#293681] hover:text-[#293681] font-medium">
                ดูคำแนะนำสไตล์ <i class="fa-solid fa-chevron-down text-[10px]" id="chevron-${shape.id}"></i>
            </button>
            <div id="detail-${shape.id}" class="hidden pt-3 border-t border-slate-100 space-y-2 text-xs text-slate-600">
                <p><span class="font-semibold text-slate-800">ทรงผมชาย:</span> ${shape.hair.male}</p>
                <p><span class="font-semibold text-slate-800">ทรงผมหญิง:</span> ${shape.hair.female}</p>
                <p><span class="font-semibold text-slate-800">แว่นตา:</span> ${shape.glass}</p>
                <p><span class="font-semibold text-slate-800">เครื่องประดับ:</span> ${shape.jewelry}</p>
            </div>`;
        grid.appendChild(card);
    });
}

export function toggleGuideDetail(id) {
    document.getElementById(`detail-${id}`).classList.toggle("hidden");
    document.getElementById(`chevron-${id}`).classList.toggle("rotate-180");
}
