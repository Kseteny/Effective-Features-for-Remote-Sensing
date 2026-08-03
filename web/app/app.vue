<script setup lang="ts">
const { title, load } = useDataset()

// Название датасета подтягиваем только в браузере: иначе сборка
// статических страниц документации потребовала бы запущенного
// сервера расчётов, а он к ним отношения не имеет.
onMounted(() => load())
</script>

<template>
  <div>
    <div class="bg-deco bg-deco--left" aria-hidden="true">
      <svg viewBox="0 0 340 900" preserveAspectRatio="none">
        <defs>
          <linearGradient id="bgFade" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stop-color="#fff" stop-opacity="1" />
            <stop offset="0.7" stop-color="#fff" stop-opacity="0.5" />
            <stop offset="1" stop-color="#fff" stop-opacity="0" />
          </linearGradient>
          <mask id="bgFadeMask">
            <rect width="340" height="900" fill="url(#bgFade)" />
          </mask>
        </defs>
        <g class="bg-deco__lines" mask="url(#bgFadeMask)">
          <path d="M -20 40 C -1 26, 17 46, 37 33 C 55 41, 74 34, 93 33 C 112 40, 131 21, 150 39 C 169 22, 187 23, 207 39 C 225 53, 244 25, 263 35 C 282 45, 301 58, 320 41" opacity="0.5" />
          <path d="M -20 124 C -1 93, 17 148, 37 118 C 55 100, 74 98, 93 119 C 112 145, 131 102, 150 126 C 169 133, 187 115, 207 125 C 225 95, 244 94, 263 116 C 282 136, 301 119, 320 119" opacity="0.49" />
          <path d="M -20 214 C -1 205, 17 228, 37 218 C 55 203, 74 218, 93 215 C 112 232, 131 225, 150 210 C 169 237, 187 197, 207 213 C 225 226, 244 198, 263 214 C 282 193, 301 222, 320 219" opacity="0.48" />
          <path d="M -20 304 C -1 293, 17 317, 37 307 C 55 309, 74 302, 93 313 C 112 332, 131 303, 150 309 C 169 277, 187 317, 207 308 C 225 336, 244 325, 263 299 C 282 297, 301 315, 320 292" opacity="0.46" />
          <path d="M -20 391 C -1 377, 17 375, 37 394 C 55 378, 74 382, 93 389 C 112 403, 131 376, 150 390 C 169 392, 187 404, 207 395 C 225 403, 244 383, 263 389 C 282 386, 301 404, 320 397" opacity="0.45" />
          <path d="M -20 466 C -1 456, 17 457, 37 466 C 55 469, 74 458, 93 459 C 112 463, 131 461, 150 467 C 169 482, 187 473, 207 466 C 225 470, 244 472, 263 460 C 282 480, 301 476, 320 471" opacity="0.44" />
          <path d="M -20 564 C -1 559, 17 546, 37 566 C 55 545, 74 545, 93 559 C 112 549, 131 557, 150 556 C 169 542, 187 549, 207 557 C 225 558, 244 543, 263 570 C 282 569, 301 548, 320 559" opacity="0.43" />
          <path d="M -20 646 C -1 630, 17 661, 37 654 C 55 645, 74 645, 93 639 C 112 629, 131 639, 150 642 C 169 660, 187 632, 207 638 C 225 665, 244 647, 263 640 C 282 648, 301 626, 320 646" opacity="0.42" />
          <path d="M -20 750 C -1 762, 17 735, 37 747 C 55 729, 74 767, 93 751 C 112 768, 131 740, 150 743 C 169 770, 187 781, 207 759 C 225 769, 244 770, 263 756 C 282 733, 301 751, 320 747" opacity="0.4" />
          <path d="M -20 821 C -1 815, 17 814, 37 823 C 55 835, 74 820, 93 826 C 112 835, 131 834, 150 820 C 169 813, 187 813, 207 818 C 225 813, 244 825, 263 826 C 282 831, 301 821, 320 823" opacity="0.39" />
        </g>
      </svg>
    </div>
    <div class="bg-deco bg-deco--right" aria-hidden="true">
      <svg viewBox="0 0 340 900" preserveAspectRatio="none">
        <defs>
          <linearGradient id="bgFadeR" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stop-color="#fff" stop-opacity="1" />
            <stop offset="0.7" stop-color="#fff" stop-opacity="0.5" />
            <stop offset="1" stop-color="#fff" stop-opacity="0" />
          </linearGradient>
          <mask id="bgFadeMaskR">
            <rect width="340" height="900" fill="url(#bgFadeR)" />
          </mask>
        </defs>
        <g class="bg-deco__lines" mask="url(#bgFadeMaskR)">
          <path d="M -20 80 C -1 60, 17 95, 37 70 C 55 50, 74 88, 93 65 C 112 45, 131 92, 150 68 C 169 100, 187 55, 207 78 C 225 62, 244 96, 263 71 C 282 48, 301 84, 320 66" opacity="0.5" />
          <path d="M -20 176 C -1 160, 17 190, 37 168 C 55 200, 74 155, 93 178 C 112 162, 131 196, 150 172 C 169 150, 187 184, 207 165 C 225 198, 244 160, 263 182 C 282 168, 301 192, 320 170" opacity="0.48" />
          <path d="M -20 268 C -1 288, 17 250, 37 275 C 55 258, 74 292, 93 268 C 112 250, 131 284, 150 265 C 169 296, 187 255, 207 278 C 225 262, 244 290, 263 270 C 282 252, 301 280, 320 266" opacity="0.46" />
          <path d="M -20 372 C -1 352, 17 386, 37 366 C 55 398, 74 356, 93 380 C 112 360, 131 392, 150 370 C 169 350, 187 384, 207 364 C 225 396, 244 358, 263 378 C 282 362, 301 388, 320 368" opacity="0.44" />
          <path d="M -20 470 C -1 490, 17 452, 37 476 C 55 458, 74 494, 93 470 C 112 452, 131 486, 150 466 C 169 498, 187 456, 207 480 C 225 462, 244 492, 263 472 C 282 454, 301 484, 320 468" opacity="0.42" />
          <path d="M -20 574 C -1 554, 17 588, 37 568 C 55 600, 74 558, 93 582 C 112 562, 131 594, 150 572 C 169 552, 187 586, 207 566 C 225 598, 244 560, 263 580 C 282 564, 301 590, 320 570" opacity="0.4" />
          <path d="M -20 672 C -1 692, 17 654, 37 678 C 55 660, 74 696, 93 672 C 112 654, 131 688, 150 668 C 169 700, 187 658, 207 682 C 225 664, 244 694, 263 674 C 282 656, 301 686, 320 670" opacity="0.39" />
          <path d="M -20 780 C -1 760, 17 794, 37 774 C 55 806, 74 764, 93 788 C 112 768, 131 800, 150 778 C 169 758, 187 792, 207 772 C 225 804, 244 766, 263 786 C 282 770, 301 796, 320 776" opacity="0.37" />
        </g>
      </svg>
    </div>

    <header class="site-head">
      <div class="site-head__inner">
        <NuxtLink to="/run" class="brand">
          <svg class="brand__mark" width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
            <rect x="0" y="0" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.85" />
            <rect x="6.5" y="0" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.5" />
            <rect x="13" y="0" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.3" />
            <rect x="0" y="6.5" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.5" />
            <rect x="6.5" y="6.5" width="5" height="5" rx="1" fill="var(--gold)" />
            <rect x="13" y="6.5" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.3" />
            <rect x="0" y="13" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.3" />
            <rect x="6.5" y="13" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.5" />
            <rect x="13" y="13" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.85" />
          </svg>
          <div class="brand__text">
            <div class="brand__name">Отбор признаков</div>
            <div class="brand__sub">{{ title }}</div>
          </div>
        </NuxtLink>

        <nav class="site-nav">
          <NuxtLink to="/run">Расчёт</NuxtLink>
          <NuxtLink to="/history">История</NuxtLink>
          <NuxtLink to="/data">Данные</NuxtLink>
          <NuxtLink to="/">Признаки</NuxtLink>
          <span class="site-nav__gap"></span>
          <NuxtLink to="/docs">О проекте</NuxtLink>
          <NuxtLink to="/docs/methods">Методы</NuxtLink>
        </nav>
      </div>
    </header>

    <main>
      <NuxtPage />
    </main>
  </div>
</template>
