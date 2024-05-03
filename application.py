import streamlit as st
import pandas as pd
from src.pipelines.prediction_pipeline import predict
import warnings
warnings.filterwarnings("ignore")



def main():

    st.title("Обнаружение мошенничества с кредитными картами")

   
    st.image('https://images.pexels.com/photos/259200/pexels-photo-259200.jpeg',
              use_column_width=True)

    
    uploaded_file = st.file_uploader("Загрузите CSV файл", label="Перетащите сюда файл", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Загруженные данные")
        st.write(df)

        predictions = predict(df)
        final = pd.concat([df['ID'], pd.Series(predictions).map(
            {0: 'ok', 1: 'fraud'})], axis=1)

        st.subheader("Предсказания")
        st.write(final)



if __name__ == '__main__':
    main()