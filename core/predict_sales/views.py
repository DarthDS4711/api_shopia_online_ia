from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
from sklearn.linear_model import LinearRegression

from artificial_inteligence_classes.LinnearRegression import LinnearRegression

# api to make a linear regression prediction for sales like next day, next month, etc
class PredictedNumberSalesApi(APIView):

    def get(self, request, format=None):
        data = request.data
        response = {}
        if ('data_x' in data) == False:
            return Response({'error' : 'data_x can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        if ('data_y' in data) == False:
            return Response({'error' : 'data_y can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        if ('data_predict' in data) == False:
            return Response({'error' : 'data_predict can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            print("-------------------------------INCOMING DATA------------------------------")
            print(data['data_x'])
            print(data['data_y'])
            print(data['data_predict'])
            data_x = np.array(data['data_x'], dtype=np.float64)
            data_y = np.array(data['data_y'], dtype=np.float64)
            linnear_regresion = LinnearRegression()
            w, b = linnear_regresion.train_linnear_regression(data_x, data_y)
            prediction = linnear_regresion.compute_model(w, b, data['data_predict'])
            response['data_predicted'] = round(prediction, 2)
            response['w'] = w
            response['b'] = b
            print(response)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            response['error'] = str(e)
        return Response(response, status=status.HTTP_500_INTERNAL_SERVER_ERROR)