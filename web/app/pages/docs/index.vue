<script setup lang="ts">
useHead({ title: 'О проекте — Отбор признаков' })
</script>

<template>
  <article class="doc">
    <h1>О проекте</h1>

    <p>
      Инструмент сравнивает методы отбора признаков в задачах дистанционного
      зондирования Земли. По мультиспектральному снимку вычисляется набор
      признаков, затем разные критерии отбирают из них наиболее информативные —
      а инструмент показывает, насколько результаты этих критериев совпадают
      и чего стоит каждый по времени.
    </p>

    <h2>Данные</h2>
    <p>
      Используется датасет MultiSenGE — снимки Sentinel-2 с разметкой
      землепользования по региону Гранд-Эст во Франции. Всего 1911 фрагментов
      размером 256×256 пикселей, каждый содержит 10 спектральных каналов
      и попиксельную разметку на 14 классов.
    </p>
    <p>
      Классы сильно несбалансированы: три крупнейших — пахотные земли, леса
      и луга — занимают около 90% размеченной площади, а самые редкие
      встречаются менее чем на одном проценте. Это влияет на результаты,
      поэтому наряду с точностью считается F1-macro, где все классы
      учитываются одинаково.
    </p>

    <h2>Признаки</h2>
    <p>
      Всего вычисляется 41 признак: 9 спектральных и 32 текстурных.
      Спектральные — шесть нормализованных каналов Sentinel-2 и три индекса
      (NDVI, NDWI, NDBI). Текстурные считаются в четырёх скользящих окнах:
      3×3, 5×5, 7×7 и 9×9, по восемь признаков на окно.
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

<style scoped>
.doc { padding: 2rem; max-width: 720px; line-height: 1.65; }
h1 { margin-top: 0; }
h2 { margin-top: 2rem; font-size: 1.15rem; }
li { margin: .5rem 0; }
</style>