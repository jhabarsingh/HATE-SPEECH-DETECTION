<template>
  <v-parallax
    dark
    src="https://cdn.vuetifyjs.com/images/backgrounds/vbanner.jpg"
    style="min-height:100vh;"
  >
    <v-row
      align="center"
      justify="center"
    >
      <v-col class="text-center" cols="12">
        <HateSpeechChild />
      </v-col>
    </v-row>
  </v-parallax>
</template>

<script>
import HateSpeechChild from './HateSpeechChild.vue'
export default {
  data: () => ({
  }),
  components: {
    HateSpeechChild
  }
}
</script>