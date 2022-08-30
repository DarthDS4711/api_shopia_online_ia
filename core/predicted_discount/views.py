from array import array
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np

from artificial_inteligence_classes.MultilayerPerceptron import MultilayerPerceptron

# api to make a linear regression prediction for sales like next day, next month, etc
class PredictedDiscountProductsApi(APIView):

    def get(self, request, format=None):
        data = request.data
        response = {}
        if ('data_class1_X1' in data) == False:
            return Response({'error' : 'data_class1_X1 can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        if ('data_class1_X2' in data) == False:
            return Response({'error' : 'data_class1_X2 can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        if ('data_class2_X1' in data) == False:
            return Response({'error' : 'data_class2_X1 can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        if ('data_class2_X2' in data) == False:
            return Response({'error' : 'data_class2_X2 can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        if ('data_class3_X1' in data) == False:
            return Response({'error' : 'data_class3_X1 can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        if ('data_class3_X2' in data) == False:
            return Response({'error' : 'data_class3_X2 can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        if ('data_predict' in data) == False:
            return Response({'error' : 'data_predict can not be empty or null'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            print("-------------------------------INCOMING DATA------------------------------")
            print(data['data_class1_X1'])
            print(data['data_class2_X1'])
            print(data['data_class3_X1'])
            print(data['data_class1_X2'])
            print(data['data_class2_X2'])
            print(data['data_class3_X2'])
            print(data['data_predict'])
            response = {}
            multilayer_percetron = MultilayerPerceptron()
            multilayer_percetron.set_data_for_train(data['data_class1_X1'], data['data_class1_X2'],
            data['data_class2_X1'], data['data_class2_X2'], data['data_class3_X1'], data['data_class1_X2'])
            multilayer_percetron.set_n_hidden_neurons_hidden_layers(10, 5)
            multilayer_percetron.set_random_weigths()
            multilayer_percetron.train_net()
            data_predict = np.array(data['data_predict'])
            predicted_discount = multilayer_percetron.predict(data_predict)
            print(predicted_discount)
            response['class'] = predicted_discount[0][0]
            print(response)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            response['error'] = str(e)
        return Response(response, status=status.HTTP_500_INTERNAL_SERVER_ERROR)