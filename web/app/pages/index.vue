<script setup lang="ts">
import type { FeaturesResponse } from '~/types/api'

const config = useRuntimeConfig()
const { data, pending, error } = await useFetch<FeaturesResponse>(
  `${config.public.apiBase}/api/features`
)
</script>

<template>
  <div style="font-family: sans-serif; padding: 2rem; max-width: 900px;">
    <h1>Признаки</h1>

    <p v-if="pending">Загружаю…</p>

    <p v-else-if="error" style="color: crimson">
      Не получилось: {{ error.message }}
    </p>

    <template v-else-if="data">
      <p>
        Всего {{ data.total }}:
        спектральных {{ data.spectral }},
        текстурных {{ data.textural }}
      </p>
      <ul>
        <li v-for="f in data.items" :key="f.index">
          <b>#{{ f.index }} {{ f.name }}</b> — {{ f.description }}
        </li>
      </ul>
    </template>
  </div>
</template>