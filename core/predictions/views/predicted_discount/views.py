from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class TestClassApi(APIView):
    def get(self, request, format=None):
        data = {}
        data['message'] = 'hello'
        return Response(data, status=status.HTTP_202_ACCEPTED)