<script setup lang="ts">
useHead({ title: 'О проекте' })
</script>

<template>
  <article class="page page--narrow prose">
    <PageHeader>
      О проекте
      <template #lead>
        Инструмент сравнивает методы отбора признаков в задачах
        дистанционного зондирования Земли.
      </template>
    </PageHeader>

    <p>
      По мультиспектральному снимку вычисляется набор признаков, затем разные
      критерии отбирают из них наиболее информативные. Инструмент показывает,
      насколько результаты этих критериев совпадают и чего стоит каждый
      по времени.
    </p>

    <h2>Данные</h2>
    <p>
      Используется датасет MultiSenGE - снимки Sentinel-2 с разметкой
      землепользования по региону Гранд-Эст во Франции. Всего
      <span class="num">1911</span> фрагментов размером
      <span class="num">256×256</span> пикселей, каждый содержит
      <span class="num">10</span> спектральных каналов и попиксельную
      разметку на <span class="num">14</span> классов.
    </p>
    <p class="prose__note">
      Классы сильно несбалансированы: три крупнейших - пахотные земли, леса
      и луга - занимают около 90% размеченной площади, а самые редкие
      встречаются менее чем на одном проценте. Поэтому наряду с точностью
      считается F1-macro, где все классы учитываются одинаково.
    </p>

    <h2>Признаки</h2>
    <p>
      Всего вычисляется <span class="num">41</span> признак:
      <span class="num">9</span> спектральных и <span class="num">32</span>
      текстурных. Спектральные - шесть нормализованных каналов Sentinel-2
      и три индекса (NDVI, NDWI, NDBI). Текстурные считаются в четырёх
      скользящих окнах: 3×3, 5×5, 7×7 и 9×9, по восемь признаков на окно.
      Полный список - на странице <NuxtLink to="/">«Признаки»</NuxtLink>.
    </p>

    <h2>Критерии отбора</h2>
    <p class="tag-row">
      <span class="tag tag--criterion" style="background: var(--forest)">Бхаттачарья</span>
      <span class="tag tag--criterion" style="background: var(--plum)">Махаланобис</span>
      <span class="tag tag--criterion" style="background: var(--water)">Взаимная информация</span>
      <span class="tag tag--criterion" style="background: var(--gold)">Отбор через kNN</span>
    </p>
    <p>
      Первые три не зависят от какой-либо модели и считаются за секунды,
      последний обучает классификатор и работает заметно дольше.
    </p>
    <p>
      Смысл сравнения в том, чтобы понять, можно ли обойтись быстрым
      критерием, не привязанным к конкретному классификатору. Подробнее -
      на странице <NuxtLink to="/docs/methods">«Методы»</NuxtLink>.
    </p>

    <h2>Источники</h2>
    <ul>
      <li>
        Wenger R., Puissant A., Weber J., Idoumghar L., Forestier G.
        MultiSenGE: a Multimodal and Multitemporal Benchmark Dataset for
        Land Use/Land Cover Remote Sensing Applications. ISPRS Annals, 2022.
      </li>
      <li>
        Wenger R. et al. Multitemporal and Multimodal Deep Learning for
        Land Use/Land Cover Classification. Remote Sensing, 2023.
      </li>
    </ul>
  </article>
</template>