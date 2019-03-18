#-*-coding:utf-8-*-
import numpy as np # linear algebra
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn import preprocessing
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
from Excel_Control1 import *
from keraspp import skeras
import math, time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from matplotlib import rcParams, style
plt.rcParams['font.family'] = 'HYGungSo-Bold'

# dataset
class LSTM_Each:
    def __init__(self, symbol_name):
        self.price_dataset = 0
        self.symbol_name = symbol_name
        self.stock_price_close = 0
        self.stock_price_open = 0
        self.stock_price_high = 0
        self.stock_price_low = 0
        self.stock_price_volume = 0
        self.stock_price_total=0
        self.trainX = 0
        self.trainY =0
        self.testX =0
        self.testY =0
        self.model = 0
        self.min_max_scaler_open = preprocessing.MinMaxScaler()
        self.min_max_scaler_close = preprocessing.MinMaxScaler()
        self.min_max_scaler_low = preprocessing.MinMaxScaler()
        self.min_max_scaler_high = preprocessing.MinMaxScaler()
        self.min_max_scaler_volume = preprocessing.MinMaxScaler()
        self.score_evalutation = pd.DataFrame()
        self.score_evalutation_new = pd.DataFrame()

    def get_score_evalutation(self):
        return self.score_evalutation

    def get_price_dataset(self):
        return self.score_evalutation

    def set_price_dataset(self, price_dataset):
        self.price_dataset = price_dataset

    def set_trainX(self, trainX):
        self.trainX = trainX
        print(self.trainX.shape)
    def set_trainY(self, trainY):
        self.trainY = trainY
    def set_testX(self, testX):
        self.testX = testX
    def set_testY(self, testY):
        self.testY = testY

    def normalize_data(self, df):
        df['open'] = self.min_max_scaler_open.fit_transform(df.open.values.reshape(-1, 1))
        df['low'] = self.min_max_scaler_low.fit_transform(df.low.values.reshape(-1, 1))
        df['high'] = self.min_max_scaler_high.fit_transform(df.high.values.reshape(-1, 1))
        df['volume'] = self.min_max_scaler_volume.fit_transform(df.volume.values.reshape(-1, 1))
        df['close'] = self.min_max_scaler_close.fit_transform(df.close.values.reshape(-1, 1))
        return df

    def model_score(model, X_train, y_train, X_test, y_test):
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
        return trainScore[0], testScore[0]

    # 데이터 준비
    def load_data(self, stock, seq_len):
        amount_of_features = len(stock.columns)  # 5
        data = stock.as_matrix()
        sequence_length = seq_len + 1  # index starting from 0
        result = []

        for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
            result.append(data[index: index + sequence_length])  # index : index + 22days

        result = np.array(result)
        row = round(0.9 * result.shape[0])  # 90% split
        train = result[:int(row), :]  # 90% date, all features

        x_train = train[:, :-1]             # 마지막 열인 close를 제외한 값만 추출
        y_train = train[:, -1][:, -1]       # 마지막 열인 close만 추출

        x_test = result[int(row):, :-1]         # 마지막 열인 close를 제외한 값만 추출
        y_test = result[int(row):, -1][:, -1]   # 마지막 열인 close만 추출
        #print('y_test:{}'.format(y_test))

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
        #print('x_test:{}'.format(x_test))

        return [x_train, y_train, x_test, y_test]

# Step 2 Build Model
    def build_model(self, layers):
        d = 0.3
        model = Sequential()

        model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))

        model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))

        model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
        model.add(Dense(1, kernel_initializer="uniform", activation='linear'))

        # adam = keras.optimizers.Adam(decay=0.2)

        start = time.time()
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print("Compilation Time : ", time.time() - start)

        return model

    def denormalize(self, normalized_value):
        #df = df.close.values.reshape(-1, 1)
        normalized_value = normalized_value.reshape(-1, 1)
        new = self.min_max_scaler_close.inverse_transform(normalized_value)
        return new

    def model_score(self, model, X_train, y_train, X_test, y_test,symbol_name):
        print("symbol_name:",symbol_name)
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

        return trainScore[0], testScore[0]


if __name__ == "__main__":

    # excel 읽어오기
    dir = "/home/inhyunlee/바탕화면/Stock_Data"
    excel_obj = Excel_Control_Multi()
    excel_obj.set_dir(dir)
    excel_obj.set_path_excel_file_folder()  # dir 하위 폴더명을 검색하여 활용하여 폴더명 리스트 생성
    folders_name = excel_obj.get_folders_name()  # 폴더 이름 리스트 얻기
    excel_obj.set_combined_path_file()  # 폴더명까지의 경로를 생성함. 파일이 없을때 쓰기를 위함.
    excel_obj.set_combined_path_file_without_filename(folders_name)  # 폴더안의 파일읽기나 파일이 있을 때 쓰기를 위함
    print(excel_obj.get_folders_name())

    xlsx_file_fullpaths = excel_obj.get_combined_path_file()  # 파일명까지의 전체 경로를 얻어서 리스트로 저장

    # flag는 폴더명이 있는지를 검토하는 flag
    flag = 0

    # count는 해당 폴더의 위치에 대한 값을 저장하기위한 기준값
    count = 0

    train_total = []
    test_total = []

    windows = [1, 5, 10, 22]

    for window in windows:
        for folder_name in folders_name:
            flag = 1
            count = count + 1
            # 대신증권 종목과 같은 폴더 발견여부 확인
            # print("같다:{}".format(name))
            # print("count:{}".format(count))
            # print(xlsx_file_fullpaths[count-1])
            temp_dir = (dir + '/' + folder_name + '/' + '{}' + '.xlsx').format(folder_name)
            excel_obj.read_excel_file(temp_dir)
            price_dataset =excel_obj.dataframe
            price_dataset = price_dataset[['open', 'low', 'high', 'volume','close']]
            price_dataset_copy = price_dataset.copy(deep=True)  # deep copy를 하여 원본 데이터를 보존한다.
            symbol_name = folder_name
            print(symbol_name)


            # 1000개 이상의 기록을 보유한 종목만 학습
            if (len(price_dataset)>1000):

                window = 1
                # 입력 X=t 대한 출력 Y=t+1로 reshape
                lstm_obj = LSTM_Each(symbol_name)
                lstm_obj.set_price_dataset(price_dataset_copy)
                price_dataset_norm = lstm_obj.normalize_data(lstm_obj.price_dataset)
                #print('price_dataset_norm:{}'.format(price_dataset_norm))

                X_train, y_train, X_test, y_test = lstm_obj.load_data(price_dataset_norm, window)
                #print(X_train[0], y_train[0]) # 22일의 time series의 X_train과 22일째의 y_train값 한 세트

                #print('X_train_shape:{}'.format(X_train.shape))
                #print('y_train_shape:{}'.format(y_train.shape))
                #print('X_test_shape:{}'.format(X_test.shape))
                #print('y_test_shape:{}'.format(y_test.shape))

                #print(X_train[0], y_train[0])

                model = lstm_obj.build_model([5, window, 1])
                h = model.fit(X_train, y_train, batch_size=512, epochs=30, validation_split=0.1, verbose=1)

                fig, ax = plt.subplots(nrows=1, ncols=1)
                skeras.plot_loss(h)
                plt.title('History of training')
                #plt.show()
                #skeras.save_history_history(symbol_name, h, fold='/home/deep-core2/바탕화면/Stock1')
                #fig.savefig('/home/deep-core2/바탕화면/Stock1/{}_training_history.png'.format(symbol_name))

                diff = []
                ratio = []
                predicted_test_norm = model.predict(X_test)  # test set의 X(normalized)값에 대해서 예측한 y(normalized)값을 p라고 명함.
                print(predicted_test_norm.shape)
                # for each data index in test data
                for u in range(len(y_test)):   # y_test의 날짜 만큼 예측 값을 기록함.
                    # pr = prediction day u
                    pr = predicted_test_norm[u][0]
                    # (y_test day u / pr) - 1
                    ratio.append((y_test[u] / pr) - 1)      # 비율
                    diff.append(abs(y_test[u] - pr))        # 차이
                    #print("여기", u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))
                # Last day prediction
                #print(predicted_test_norm[-1])

                #df = pd.read_csv("../input/prices-split-adjusted.csv", index_col=0)
                #df["adj close"] = df.close  # Moving close to the last column
                #df.drop(['close'], 1, inplace=True)  # Moving close to the last column
                #df = df[df.symbol == 'GOOG']
                #df.drop(['symbol'], 1, inplace=True)

                #newp = lstm_obj.denormalize(price_dataset_norm, p)
                #newy_test = lstm_obj.denormalize(price_dataset_norm, y_test)

                predicted_test = lstm_obj.denormalize(predicted_test_norm)
                newy_test = lstm_obj.denormalize(y_test)
                #print(newy_test)

                #print('predicted_test:{}'.format(predicted_test))
                #print('newy_test:{}'.format(newy_test))

                train, test = lstm_obj.model_score(model, X_train, y_train, X_test, y_test, folder_name)

                train_total.append(train)
                test_total.append(test)

                #print(train_total)
                #print(test_total)
                total_dataframe = {'train_score': train_total, 'test_score': test_total}

                total_dataframe = pd.DataFrame(total_dataframe)
                print(total_dataframe)

                temp_dir_score = "/home/inhyunlee/바탕화면/Stock_Data_Result/{}.xlsx".format(window)
                writer = pd.ExcelWriter(temp_dir_score, engine='xlsxwriter')
                total_dataframe.to_excel(writer, 'Sheet1', index=False)
                writer.save()

                fig, ax = plt.subplots(nrows=1, ncols=1)
                #ax.set_title(folder_name)
                ax.plot(predicted_test, color='red', label='Prediction')
                ax.plot(newy_test, color='blue', label='Actual')
                ax.legend(loc='best')
                #plt.show()
                fig.savefig('/home/inhyunlee/바탕화면/Stock_Data_Result//{}/{}.png'.format(window, symbol_name))

