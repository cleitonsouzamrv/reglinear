import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Regressão Linear")

# Upload do arquivo
arquivo = st.file_uploader("Envie um arquivo CSV ou Excel", type=["csv", "xlsx"])

if arquivo is not None:
    # Leitura da base
    if arquivo.name.endswith('.csv'):
        dados = pd.read_csv(arquivo, sep=None, engine='python')
    else:
        dados = pd.read_excel(arquivo)

    st.subheader("Prévia dos dados")
    st.dataframe(dados.head(10))

    colunas = dados.columns.tolist()
    col_x = st.selectbox("Selecione a variável independente (X)", colunas)
    col_y = st.selectbox("Selecione a variável dependente (y)", colunas)

    if col_x != col_y:
        X = dados[[col_x]]
        y = dados[col_y]

        modelo = LinearRegression().fit(X, y)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Tabela de Dados")
            st.table(dados[[col_x, col_y]].head(10))

        with col2:
            st.header("Gráfico de Dispersão")
            fig, ax = plt.subplots()
            ax.scatter(X, y, color='blue')
            ax.plot(X, modelo.predict(X), color='red')
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            st.pyplot(fig)

        st.header(f"Novo valor para {col_x}:")
        novo_valor = st.number_input("Insira novo valor", min_value=1.0, max_value=999999.0, value=1500.0, step=5.0)
        processar = st.button("Processar")

        if processar:
            dados_novo_valor = pd.DataFrame([[novo_valor]], columns=[col_x])
            prev = modelo.predict(dados_novo_valor)
            st.success(f"Previsão de {col_y}: {prev[0]:.2f}")
    else:
        st.warning("Selecione colunas diferentes para X e y")
else:
    st.info("Aguardando o upload de um arquivo de dados.")
