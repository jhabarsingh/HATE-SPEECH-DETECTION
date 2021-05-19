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
       Hate Speech Detector
      </h3>
      <div class="text--primary">
          I Won't Let You Send Hatespeech
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
            color="primary"
            class="ma-2 white--text"
            @click="$router.push('/')"
            >
            Home
            <v-icon right dark>mdi-home</v-icon>
            </v-btn>

            <v-spacer />
            
            <v-btn
            :loading="loading"
            :disabled="loading"
            color="blue-grey"
            class="ma-2 white--text"
            @click="send"
            >
            Send
            <v-icon right dark>mdi-send</v-icon>
            </v-btn>
         </div>

        </v-col>

      </v-row>
    </v-container>
  </v-form>
  <Dialog :dialog=dialog :data=data />
</v-card>
</template>

<script>
  import Dialog from './Dialog.vue';

  export default {
    components: {
        Dialog,
    },
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
      send() {
          this.loader = 'loading'
          this.dialog = true
          this.backend();
      },
      backend() {
            const data = { text: this.message };
            const url = "http://localhost:8000/hatespeech/"
            fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
            .then(response => response.json())
            .then(data => {
                console.log('Success:', data);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
      }
    },
  }
</script>
