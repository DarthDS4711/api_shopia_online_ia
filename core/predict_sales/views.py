from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
from sklearn.linear_model import LinearRegression

# api to make a linear regression prediction for sales like next day, next month, etc
class PredictedNumberSalesApi(APIView):

    def get(self, request, format=None):
        data = request.data
        response = {}
        if ('list' in data) == False:
            return Response({'error' : 'List can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        if ('data_predict' in data) == False:
            return Response({'error' : 'data_predict can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            array_x = np.array(0)
            array_y = np.array(0)
            for element in data['list']:
                array_x = np.append(array_x, element[0])
                array_y = np.append(array_y, element[1])
            lineal_regresion = LinearRegression()
            response = {}
            lineal_regresion.fit(array_x.reshape(-1, 1), array_y)
            response['w'] = str(lineal_regresion.coef_)
            response['b'] = str(lineal_regresion.intercept_)
            prediction = lineal_regresion.predict(np.array(data['data_predict']).reshape(-1, 1))
            response['prediction'] = str(round(prediction[0], 2))
            return Response(response, status=status.HTTP_200_OK)
        except:
            response['error'] = 'An internal error ocurred please call admin'
        return Response(response, status=status.HTTP_500_INTERNAL_SERVER_ERROR)