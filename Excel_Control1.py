# 엑셀파일로 관리하는 함수 (전 종목 관리)
# Date : 2018년 2월 12일
# Author : 이인현
# 엑셀과 관련한 처리를 위한 클래스.
# 기능1: 엑셀을 읽어나 쓰기를 실행할 경로를 설정한다.
# 기능2: 경로안에 저장되어 있는 파일 혹은 폴더명을 조회한다.
# 기능3: 경로명에 엑셀명(종목명.xlsx)을 결합하여 전체 경로를 완성한다.
# 기능4: 엑셀을 읽거나 저장한다.

import pandas as pd
import os
#from Cyboss_Control1 import *

# 엑셀을 읽어 오거나, 저장하는 함수
class Excel_Control_Multi:
    def __init__(self):
        self.dir = dir
        self.folders_name = [] # 각 종목의 폴더명 저장 (=종목명)
        self.xlsx_file_fullpaths_without_filename = [] # 엑셀파일명을 제외한 경로명 "C:/Users/GIGABYTE/Desktop/StockHistory2/IBKS제3호스팩"
        self.xlsx_file_fullpaths = [] # dir, 폴더명, 파일명을 결합하여 전체 경로를 완성하는 함수 (엑셀파일명(업종명))
        self.stockname_list = []  # 종목명 리스트
        self.stockname = 0  # 종목명
        self.dataframe = pd.DataFrame() # dataframe

    # 초기 경로 설정
    def set_dir(self, dir):
        self.dir = dir

    # folder_name 리스트 얻기
    def get_folders_name(self):
        return self.folders_name

    # 종목명 설정
    def set_stock_name(self, stock_name):
        self.stockname = stock_name
        print(self.stockname)

    # 데이터프레임 설정
    def set_dataframe(self, dataframe):
        self.dataframe = dataframe
        print(self.dataframe)

    # 종목명 리턴
    def get_stock_name(self):
        return self.stockname

    # 데이터프레임 리턴
    def get_dataframe(self):
        return self.dataframe

    # 종목별 전체 경로 리턴 (리스트)
    def get_combined_path_file(self):
        return self.xlsx_file_fullpaths

    def get_xlsx_file_fullpaths_without_filename(self):
        return self.xlsx_file_fullpaths_without_filename

    #1. dir 경로안에 저장되어 있는 모든 파일 및 폴더명을 검색하여 리스트로 저장하는 함수 (폴더명 = 종목명)
    def set_path_excel_file_folder(self):
        self.folders_name = os.listdir(self.dir)
        #print(self.folders_name)

    # 폴더명 안에 엑셀파일쓰기를 실행함.
    # 1.종목명 리스트를 입력받아서 폴더까지의 경로를 생성함. 엑셀명 제외
    def set_combined_path_file_without_filename(self, stockname_list):
        self.stockname_list = stockname_list
        for stock_name in self.stockname_list:
            self.xlsx_file_fullpaths_without_filename.append(self.dir + '/' + stock_name)
        #print(self.xlsx_file_fullpaths_without_filename)

    #2. dir, 폴더명, 파일명을 결합하여 전체 경로를 완성하는 함수 (엑셀명 포함. 이미 생성된 엑셀 파일 읽기)
    def set_combined_path_file(self):
        for folder in self.folders_name:
            self.xlsx_file_fullpaths.append(self.dir + '/' + folder + '/' + folder + '.xlsx')
        #print(self.xlsx_files)

    #3. xlsx_files 경로에 저장되어 있는 엑셀파일을 읽어오는 함수 (파일명(filenames)을 종목으로 활용) 1,2 이후
    def read_excel_file(self, xlsx_file_fullpath):
        self.dataframe = pd.read_excel(xlsx_file_fullpath)
        filename_split = xlsx_file_fullpath.split('/')
        filename_last = filename_split[-1]
        self.stockname = filename_last[:-5]
        #print(self.stockname)
        #print(self.dataframe)

    #4. xlsx_files 경로에 있는 데이터 프레임을 종목별로 엑셀 저장함. 해당 폴더가 없을 경우 해당폴더 생성 후 저장 1,2 이후
    def write_excel_file(self, xlsx_file_fullpaths_without_filename):
        try:
            os.mkdir(dir + '/' + '{}'.format(xlsx_file_fullpaths_without_filename))
            writer = pd.ExcelWriter(self.xlsx_file_fullpaths_without_filename, engine='xlsxwriter')
            self.dataframe.to_excel(writer, 'Sheet1', index=False)  # 저장할 데이터 프레임 세팅
            writer.save()  # 저장
        except:
            writer = pd.ExcelWriter(self.xlsx_file_fullpaths_without_filename, engine='xlsxwriter')
            self.dataframe.to_excel(writer, 'Sheet1', index=False) #저장할 데이터 프레임 세팅
            writer.save() # 저장

if __name__ == "__main__":
    pass