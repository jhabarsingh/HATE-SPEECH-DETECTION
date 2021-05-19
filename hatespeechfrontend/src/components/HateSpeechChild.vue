<template>
  <v-card
    max-width="700"
    class="mx-auto"
    style="margin-top:20px;opacity:.8;"
  >
    <v-card-text
    >
      <div>Send Message</div>
      <h3 class=" text--primary">
        Jhabar Singh Bhati
      </h3>
      <div class="text--primary">
        Please Don't Send Write Abusive Language
      </div>
    </v-card-text>
  <v-form>
    <v-container>
      <v-row>
        <v-col cols="12">
          <v-textarea
            v-model="message"
            outlined
            clearable
            label="Message"
            type="text"
          >
            <template v-slot:prepend>
            </template>
            <template v-slot:append>
              <v-fade-transition leave-absolute>
                <v-progress-circular
                  v-if="loading"
                  size="24"
                  color="info"
                  indeterminate
                ></v-progress-circular>
                <img v-else width="24" height="24" src="https://raw.githubusercontent.com/jhabarsingh/SIMADIAN/main/doc/trademark.png" alt="">
              </v-fade-transition>
            </template>
          </v-textarea>

         <div
            style="display:flex;justify-content:flex-end;"
         >
             <v-btn
            :loading="loading3"
            :disabled="loading3"
            color="blue-grey"
            class="ma-2 white--text"
            @click="loader = 'loading3'"
            >
            Send
            <v-icon right dark>mdi-send</v-icon>
            </v-btn>
         </div>

        </v-col>

      </v-row>
    </v-container>
  </v-form>
</v-card>
</template>

<script>
  export default {
    data: () => ({
      message: 'Hey!',
      loading: false,
    }),
    methods: {
      clickMe () {
        this.loading = true
        this.message = 'Wait for it...'
        setTimeout(() => {
          this.loading = false
          this.message = 'You\'ve clicked me!'
        }, 2000)
      },
    },
  }
</script>
