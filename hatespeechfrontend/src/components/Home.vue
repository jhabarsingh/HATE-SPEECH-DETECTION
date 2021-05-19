<template>
  <div>
    <v-parallax
      dark
      src="https://cdn.vuetifyjs.com/images/backgrounds/vbanner.jpg"
      style="height:100vh;position:fixed;width:100%;top:0px;"
    >
      <v-row
        align="center"
        justify="center"
      >
        <v-col
          class="text-center"
          cols="12"
        >
          <h1 class="display-1 font-weight-thick mb-4">
            HATE SPEECH DETECTOR
          </h1>

          <v-img
            width="200"
            style="margin:auto;"
            src="https://github.com/jhabarsingh/HATE-SPEECH-DETECTION/blob/main/doc/org_logo.png?raw=true"
          ></v-img>
          
          <h4 class="subheading">
            Let's Stop The Hate
          </h4>

          <v-btn
            class="ma-2"
            color="primary"
            dark
            @click="$router.push('/hatespeech')"
          >
            Home
            <v-icon
              dark
              right
            >
              mdi-home
            </v-icon>
          </v-btn>
        
        </v-col>
      </v-row>

    </v-parallax>
  </div>  
</template>

<script>
export default {

}
</script>
