from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .production import predict

class HateSpeechApi(APIView):
    '''
        HateSpeech Detection POST APi
    '''

    def post(self, request):
        try:
            text = request.data.get("text", None)
            if text == None:
                raise Exception("Invalid text")
            if len(text.strip()) == 0:
                raise Exception("Zero Size")
        except:
            data = {
                "message": "invalid Message"
            }
            return Response(data)

        text = text.strip()
        
        prediction = predict(text)
        message = "Some error has been occured, Please Try Later"

        if prediction == 0:
            message = "no"
        else:
            message = "yes"
        data = {
            "hatespeech": prediction
        }
        return Response(data)