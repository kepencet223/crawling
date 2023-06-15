import streamlit as st

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from numpy import array

import pickle

from sklearn import metrics

st.set_page_config(
    page_title="Project"
)
st.title('Finance Prediction')
st.write('Nurul Faizah (200411100174)')
st.write('Triasmi Dwi Farawati (200411100174)')
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Prepocessing","Modelling","Implementation"])

with tab1:
    st.write('Studi Kasus Finance PT.Adaro Minerals Indonesia')
    st.write('Perseroan bergerak di bidang usaha pertambangan dan perdagangan batu bara metalurgi melalui Perusahaan Anak dan menjalankan kegiatan usaha berupa jasa konsultasi manajemen. Perseroan merupakan perusahaan yang di bawah naungan AEI. Dalam menjalankan usahanya, Perseroan dan Perusahaan Anaknya didukung dengan bisnis yang terintegrasi dari tambang hingga ke stockpile dan transshipment area. ')
    
    df = pd.read_csv("bca.csv")
    st.write("Dataset Finance PT.Adaro Minerals Indonesia : ")
    st.write(df)

    st.write("Penjelasan Nama - Nama Kolom : ")
    st.write("""
    <ol>
    <li>Date (Tanggal): Tanggal dalam data time series mengacu pada tanggal tertentu saat data keuangan dikumpulkan atau dilaporkan. Ini adalah waktu kapan data keuangan yang terkait dengan PT Adaro Minerals Indonesia dicatat.</li>
    <li>Open (Harga Pembukaan): Harga pembukaan adalah harga perdagangan PT Adaro Minerals Indonesia pada awal periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga pembukaan menunjukkan harga perdagangan pertama dari PT Adaro Minerals Indonesia pada periode tersebut.</li>
    <li>High (Harga Tertinggi): Harga tertinggi adalah harga tertinggi yang dicapai oleh PT Adaro Minerals Indonesia selama periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga tertinggi mencerminkan harga perdagangan tertinggi yang dicapai oleh PT Adaro Minerals Indonesia dalam periode tersebut.</li>
    <li>Low (Harga Terendah): Harga terendah adalah harga terendah yang dicapai oleh PT Adaro Minerals Indonesia selama periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga terendah mencerminkan harga perdagangan terendah yang dicapai oleh PT Adaro Minerals Indonesia dalam periode tersebut.</li>
    <li>Close (Harga Penutupan): Harga penutupan adalah harga terakhir dari PT Adaro Minerals Indonesia pada akhir periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga penutupan menunjukkan harga terakhir di mana PT Adaro Minerals Indonesia diperdagangkan pada periode tersebut.</li>
    <li>Adj Close (Harga Penutupan yang Disesuaikan): Adj Close, atau harga penutupan yang disesuaikan, adalah harga penutupan yang telah disesuaikan untuk faktor-faktor seperti dividen, pemecahan saham, atau perubahan lainnya yang mempengaruhi harga saham PT Adaro Minerals Indonesia. Ini memberikan gambaran yang lebih akurat tentang kinerja saham dari waktu ke waktu karena menghilangkan efek dari perubahan-perubahan tersebut.</li>
    <li>Volume: Volume dalam konteks data keuangan PT Adaro Minerals Indonesia mengacu pada jumlah saham yang diperdagangkan selama periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Volume mencerminkan seberapa aktifnya perdagangan saham PT Adaro Minerals Indonesia dalam periode tersebut.</li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("Data preprocessing adalah proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur. Selain itu, data mining juga tidak dapat memproses data mentah, sehingga proses ini sangat penting dilakukan untuk mempermudah proses berikutnya, yakni analisis data.")
    st.write("Data preprocessing adalah proses yang penting dilakukan guna mempermudah proses analisis data. Proses ini dapat menyeleksi data dari berbagai sumber dan menyeragamkan formatnya ke dalam satu set data.")
    
    scaler = st.radio(
    "Pilih Metode Normalisasi Data : ",
    ('Tanpa Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        file_path = "minmax"
        with open(file_path, "rb") as file:
            minmax = pickle.load(file)
        scaler = minmax
        df_for_scaler = pd.DataFrame(df, columns = ['High','Low','Close','Adj Close','Volume'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['High','Low','Close','Adj Close','Volume'])
        df_drop_column_for_minmaxscaler=df.drop(['High','Low','Close','Adj Close','Volume'], axis=1)
        df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    st.write(df_new)

with tab3:
    st.write("""
        <h5>Modelling</h5>
        <br>
        """, unsafe_allow_html=True)
    # Menghapus baris terakhir
    df = df.drop(df.index[-1])

    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)

        return array(X), array(y)
    
    df_open = df['Open']

    n_steps = 3
    X, y = split_sequence(df_open, n_steps)

    # column names to X and y data frames
    df_X = pd.DataFrame(X, columns=['t-'+str(i+1) for i in range(n_steps-1, -1,-1)])
    df_y = pd.DataFrame(y, columns=['t (prediction)'])

    # concat df_X and df_y
    df = pd.concat([df_X,df_y], axis=1)

    #Normalisasi data
    from sklearn.preprocessing import MinMaxScaler
    scaler= MinMaxScaler()
    X_norm= scaler.fit_transform(df_X)
    y_norm= scaler.fit_transform(df_y)

    #split data train 80% test 20%
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=0)

    option = st.selectbox('Pilih Model:', ['Pilih','KNN', 'Decision Tree', 'Random Forest','Linear Regression','SVR'])
    st.write('Anda memilih:', option)
    if option == 'KNN':
        st.write('KNN')
        # Melakukan fitting dan prediksi menggunakan model KNeighborsRegressor
        from sklearn.neighbors import KNeighborsRegressor
        knn_model = KNeighborsRegressor(n_neighbors=6)
        knn_model.fit(X_train, y_train)
        knn_preds = knn_model.predict(X_test)

        #KNN
        mape = mean_absolute_percentage_error(y_test,knn_preds)
        mae = mean_absolute_error(y_test,knn_preds)
        akurasi = 1 - (mae / np.mean(y_test))
        st.write('MAPE :', mape)
        st.write('MAE :', mae)
        st.write('Akurasi:', akurasi)
        

    elif option == 'Decision Tree':
        st.write('Decision Tree')
        # Melakukan fitting dan prediksi menggunakan model DecisionTreeRegressor
        from sklearn.tree import DecisionTreeRegressor
        dt_model = DecisionTreeRegressor()
        dt_model.fit(X_train, y_train)
        dt_preds = dt_model.predict(X_test)

        #DT
        mape = mean_absolute_percentage_error(y_test,dt_preds)
        mae = mean_absolute_error(y_test,dt_preds)
        akurasi = 1 - (mae / np.mean(y_test))
        st.write('MAPE :', mape)
        st.write('MAE :', mae)
        st.write('Akurasi:', akurasi)

    elif option == 'Random Forest':
        st.write('Random Forest')
        # Melakukan fitting dan prediksi menggunakan model RandomForestRegressor
        from sklearn.ensemble import RandomForestRegressor
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)

        #RF
        mape = mean_absolute_percentage_error(y_test,rf_preds)
        mae = mean_absolute_error(y_test,rf_preds)
        akurasi = 1 - (mae / np.mean(y_test))
        st.write('MAPE :', mape)
        st.write('MAE :', mae)
        st.write('Akurasi:', akurasi)

    elif option == 'Linear Regression':
        st.write('Linear Regression')
        # Melakukan fitting dan prediksi menggunakan model LinearRegression
        from sklearn.linear_model import LinearRegression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_preds = lr_model.predict(X_test)

        #LR
        mape = mean_absolute_percentage_error(y_test,lr_preds)
        mae = mean_absolute_error(y_test,lr_preds)
        akurasi = 1 - (mae / np.mean(y_test))
        st.write('MAPE :', mape)
        st.write('MAE :', mae)
        st.write('Akurasi:', akurasi)

    elif option == 'SVR':
        st.write('SVR')
        # Melakukan fitting dan prediksi menggunakan model SVR
        from sklearn.svm import SVR
        svr_model = SVR()
        svr_model.fit(X_train, y_train)
        svr_preds = svr_model.predict(X_test)

        #SVR
        mape = mean_absolute_percentage_error(y_test,svr_preds)
        mae = mean_absolute_error(y_test,svr_preds)
        akurasi = 1 - (mae / np.mean(y_test))
        st.write('MAPE :', mape)
        st.write('MAE :', mae)
        st.write('Akurasi:', akurasi)

    else:
        st.write('Pilihan tidak valid')
            

with tab4:
    st.write("""
    <h5>Implementation Model</h5>
    <br>
    """, unsafe_allow_html=True)
    with st.form("my_form"):
        # Tambahkan inputan
        input1 = st.number_input('Input 1:')
        input2 = st.number_input('Input 2:')
        input3 = st.number_input('Input 3:')

        # Inisialisasi array kosong
        test_array = []

        # Cek apakah tombol Submit ditekan
        submitted = st.form_submit_button("Submit")

        # Jika tombol Submit ditekan, buat array dengan inputan
        if submitted:
            # Tambahkan nilai input ke dalam array
            test_array.append(input1)
            test_array.append(input2)
            test_array.append(input3)

            # Konversi array menjadi numpy array
            test_array = np.array([test_array])

            file_path = "model_LR_pkl"
            with open(file_path, "rb") as file:
                LR= pickle.load(file)
            # Lakukan prediksi dengan model Linear Regression
            LR_predict = LR.predict(test_array)

            # Tampilkan hasil
            st.success(f'Hasil Prediksi: {LR_predict.reshape(1)[0]}')

        
