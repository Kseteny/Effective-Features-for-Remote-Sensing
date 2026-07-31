<script setup lang="ts">
useHead({ title: 'Данные' })

const { report, failed, loading, load } = useDataset()

const config = useRuntimeConfig()
const { data: setsData } = await useFetch<{
  items: { id: string; name: string; description: string; builtin: boolean }[]
}>(`${config.public.apiBase}/api/feature-sets`)

onMounted(() => load(true))

const info = computed(() => (report.value?.ok ? report.value.dataset : null))

const ROLE_NAMES: Record<string, string> = {
  blue: 'синий',
  green: 'зелёный',
  red: 'красный',
  nir: 'ближний ИК',
  swir1: 'коротковолновый ИК 1',
  swir2: 'коротковолновый ИК 2',
}

/** Роль канала, если она задана в описании. */
function roleOf(band: string): string | null {
  const roles = info.value?.band_roles ?? {}
  for (const [role, b] of Object.entries(roles)) {
    if (b === band) return ROLE_NAMES[role] ?? role
  }
  return null
}

function usedInFeatures(band: string): boolean {
  return info.value?.feature_bands.includes(band) ?? false
}
</script>

<template>
  <div class="page">
    <div class="page__head">
      <h1>Данные</h1>
      <p class="page__lead">
        Что подключено сейчас и что на этом получится посчитать.
        Описание берётся из файла <code>dataset.json</code> в корне проекта.
      </p>
    </div>

    <p v-if="loading && !report" class="notice notice--empty">Читаю описание…</p>

    <p v-else-if="failed" class="notice notice--error">
      {{ failed }}. Проверьте, запущен ли он на порту 8000.
    </p>

    <div v-else-if="report && !report.ok" class="notice notice--error">
      <p class="err__title">В описании датасета ошибка</p>
      <pre class="err__text">{{ report.error }}</pre>
    </div>

    <template v-else-if="info">
      <div class="card">
        <p class="card__title">{{ info.name }}</p>
        <div class="stats">
          <div class="stats__item">
            <div class="stats__num num">{{ info.n_pairs ?? '-' }}</div>
            <div class="stats__label">пар снимок-маска</div>
          </div>
          <div class="stats__item">
            <div class="stats__num num">{{ info.n_bands }}</div>
            <div class="stats__label">каналов в файле</div>
          </div>
          <div class="stats__item">
            <div class="stats__num num">{{ info.n_classes }}</div>
            <div class="stats__label">классов</div>
          </div>
          <div class="stats__item">
            <div class="stats__num num">{{ info.n_features }}</div>
            <div class="stats__label">признаков будет</div>
          </div>
        </div>
      </div>

      <div class="card">
        <p class="card__title">Каналы</p>
        <p class="muted note">
          В признаки идут не обязательно все каналы - только те,
          что указаны в описании. Остальные читаются из файла,
          но в расчёте не участвуют.
        </p>
        <div class="bands">
          <div
            v-for="b in info.band_order"
            :key="b"
            class="band"
            :class="{ 'band--off': !usedInFeatures(b) }"
          >
            <span class="band__name">{{ b }}</span>
            <span class="band__role">{{ roleOf(b) ?? 'роль не задана' }}</span>
          </div>
        </div>
      </div>

      <div class="card">
        <p class="card__title">Спектральные индексы</p>
        <p v-if="info.indices_available.length" class="ok">
          Посчитаются: <b>{{ info.indices_available.join(', ') }}</b>
        </p>
        <p v-else class="muted">
          Ни один индекс не посчитается - не заданы роли каналов.
          Останутся только текстурные признаки.
        </p>

        <ul v-if="info.indices_missing.length" class="missing">
          <li v-for="m in info.indices_missing" :key="m.name">
            <b>{{ m.name }}</b> не посчитается - не заданы роли:
            {{ m.needs.join(', ') }}
          </li>
        </ul>
      </div>

      <div v-if="setsData?.items?.length" class="card">
        <p class="card__title">Наборы признаков</p>
        <p class="muted note">
          Свой набор можно добавить, не трогая код программы: положите файл
          в папку <code>user_features</code> рядом с проектом. Как -
          написано на странице <NuxtLink to="/docs/methods">«Методы»</NuxtLink>.
        </p>
        <div class="sets">
          <div v-for="s in setsData.items" :key="s.id" class="set">
            <div class="set__head">
              <span class="set__name">{{ s.name }}</span>
              <span v-if="!s.builtin" class="set__badge">свой</span>
            </div>
            <div class="set__desc">{{ s.description }}</div>
          </div>
        </div>
      </div>

      <div class="card">
        <p class="card__title">Классы</p>
        <p v-if="!info.classes.length" class="muted">
          Названия классов не заданы - в интерфейсе будут номера.
        </p>
        <div v-else class="classes">
          <div v-for="c in info.classes" :key="c.id" class="class-row">
            <span class="num class-row__id">{{ c.id }}</span>
            <span>{{ c.name }}</span>
          </div>
        </div>
      </div>

      <div v-if="report?.notes?.length" class="card">
        <p class="card__title">Обратите внимание</p>
        <ul class="notes">
          <li v-for="(n, i) in report.notes" :key="i">{{ n }}</li>
        </ul>
      </div>
    </template>
  </div>
</template>

<style scoped>
code {
  font-family: var(--font-mono);
  font-size: 0.88em;
  background: var(--surface-sunk);
  padding: 0.1em 0.35em;
  border-radius: var(--r-sm);
}

.stats { display: flex; flex-wrap: wrap; gap: 2.5rem; }

.stats__num {
  font-family: var(--font-display);
  font-size: 1.7rem;
  font-weight: 700;
  line-height: 1;
  color: var(--forest);
}

.stats__label { font-size: 0.78rem; color: var(--ink-faint); margin-top: 0.3rem; }

.note { font-size: 0.85rem; margin: 0 0 1rem; }

.bands { display: flex; flex-wrap: wrap; gap: 0.5rem; }

.band {
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  padding: 0.45rem 0.7rem;
  min-width: 96px;
}

.band__name {
  display: block;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  font-weight: 500;
}

.band__role { display: block; font-size: 0.7rem; color: var(--ink-faint); }

/* Канал, который не идёт в признаки: приглушён, но не спрятан -
   важно видеть, что он в файле есть. */
.band--off { background: var(--surface-sunk); opacity: 0.65; }
.band--off .band__name { font-weight: 400; }

.ok { margin: 0; }

.missing { margin: 0.8rem 0 0; padding-left: 1.1rem; color: var(--ink-soft); font-size: 0.9rem; }
.missing li { margin: 0.3rem 0; }

.sets { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0.8rem; }

.set {
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  padding: 0.7rem 0.85rem;
}

.set__head { display: flex; align-items: center; gap: 0.5rem; }
.set__name { font-weight: 500; font-size: 0.92rem; }

.set__badge {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  background: var(--forest, #14664A);
  color: #fff;
  padding: 0.1rem 0.4rem;
  border-radius: var(--r-sm);
}

.set__desc { font-size: 0.82rem; color: var(--ink-soft); margin-top: 0.3rem; line-height: 1.45; }

.classes { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 0.35rem 1.5rem; }
.class-row { display: flex; gap: 0.7rem; font-size: 0.9rem; }
.class-row__id { color: var(--ink-faint); min-width: 1.6rem; text-align: right; }

.notes { margin: 0; padding-left: 1.1rem; color: var(--ink-soft); font-size: 0.9rem; }
.notes li { margin: 0.3rem 0; }

.err__title { margin: 0 0 0.5rem; font-weight: 600; }
.err__text {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  white-space: pre-wrap;
  margin: 0;
  line-height: 1.6;
}
</style>
