<template>
  <v-row justify="center">
    <v-dialog
      v-model="dialog"
      persistent
      max-width="290"
    >
      <template>
      </template>
      <v-card>
        <v-card-title class="lamao"  >
          👋 Hate Speech Detector Here
        </v-card-title>
        <v-card-text>{{data}}</v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn
            color="green darken-1"
            text
            @click="dialog = false"
          >
            Okay
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-row>
</template>


<script>
  export default {
    props: [
        'dialog',
        'data'
    ],
    data () {
      return {
      }
    },
  }
</script>

<style scoped>

    .lamao {
        font-size: 15px;
        text-align: center;
    }
</style>