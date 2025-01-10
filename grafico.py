import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Função para salvar o gráfico como imagem
def salvar_grafico(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# Função para gerar gráficos
def gerar_grafico(tipo, categorias, valores, titulo, comparativo=None, valores_comparativos=None, nomes_conjuntos=None, eixo_y="Valores"):
    fig, ax = plt.subplots(figsize=(12, 6))

    if tipo == "Barras":
        largura = 0.2  # Largura reduzida para aumentar o espaço entre as barras
        x = np.arange(len(categorias))
        if comparativo:
            for i, valores in enumerate(valores_comparativos):
                deslocamento = (i - len(valores_comparativos) / 2) * (largura + 0.1)  # Aumentado o deslocamento
                nome_conjunto = nomes_conjuntos[i] if nomes_conjuntos else f"Conjunto {i + 1}"
                ax.bar(x + deslocamento, valores, largura, label=nome_conjunto)
                for xi, vi in zip(x + deslocamento, valores):
                    label = f"{int(vi) if isinstance(vi, float) and vi.is_integer() else vi}"
                    ax.text(xi, vi, label, ha='center', va='bottom')
        else:
            ax.bar(categorias, valores, color="skyblue")
            for i, v in enumerate(valores):
                label = f"{int(v) if isinstance(v, float) and v.is_integer() else v}"
                ax.text(i, v, label, ha='center', va='bottom')
        ax.set_xticks(x)
        ax.set_xticklabels(categorias, rotation=45, ha="right")
        ax.set_xlabel("Categorias")
        ax.set_ylabel(eixo_y)

    elif tipo == "Coluna":
        largura = 0.2
        y = np.arange(len(categorias))
        if comparativo:
            for i, valores in enumerate(valores_comparativos):
                deslocamento = (i - len(valores_comparativos) / 2) * (largura + 0.1)
                nome_conjunto = nomes_conjuntos[i] if nomes_conjuntos else f"Conjunto {i + 1}"
                ax.barh(y + deslocamento, valores, largura, label=nome_conjunto)
                for yi, vi in zip(y + deslocamento, valores):
                    label = f"{int(vi) if isinstance(vi, float) and vi.is_integer() else vi}"
                    ax.text(vi, yi, label, va='center', ha='left')
        else:
            ax.barh(categorias, valores, color="skyblue")
            for i, v in enumerate(valores):
                label = f"{int(v) if isinstance(v, float) and v.is_integer() else v}"
                ax.text(v, i, label, va='center', ha='left')
        ax.set_yticks(y)
        ax.set_yticklabels(categorias)
        ax.set_xlabel(eixo_y)
        ax.set_ylabel("Categorias")
        ax.invert_yaxis()

    elif tipo == "Pizza":
        ax.pie(valores, labels=categorias, autopct=lambda p: f"{p:.1f}%", startangle=90, colors=plt.cm.Paired.colors[:len(valores)])
        ax.axis('equal')
        ax.legend(loc="best")

    elif tipo == "Top10":
        valores_ordenados = sorted(zip(valores, categorias), reverse=True)[:10]
        valores, categorias = zip(*valores_ordenados)
        ax.barh(categorias, valores, color="skyblue")
        for i, v in enumerate(valores):
            label = f"{int(v) if isinstance(v, float) and v.is_integer() else v}"
            ax.text(v, i, label, va='center', ha='left')
        ax.set_xlabel(eixo_y)
        ax.invert_yaxis()

    elif tipo == "Linha":
        for i, valores in enumerate(valores_comparativos if comparativo else [valores]):
            nome_conjunto = nomes_conjuntos[i] if nomes_conjuntos else f"Conjunto {i + 1}"
            ax.plot(categorias, valores, marker='o', label=nome_conjunto)
            for xi, yi in zip(categorias, valores):
                label = f"{int(yi) if isinstance(yi, float) and yi.is_integer() else yi}"
                ax.text(xi, yi, label, ha='center', va='bottom')
        ax.set_xlabel("Categorias")
        ax.set_ylabel(eixo_y)

    ax.set_title(titulo, fontsize=16, fontweight='bold')
    if comparativo or tipo == "Linha":
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)

    fig.text(0.05, 0.005, 'By: Alexandre Carvalho', ha='left', va='center', fontsize=7, color='gray', alpha=0.5)
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    return salvar_grafico(fig)

# Interface do Streamlit
st.set_page_config(layout="wide", page_title="Dashboard - SIGMA BI")
st.title("Dashboard Interativo - SIGMA BI")

# Carregar a planilha
st.sidebar.header("Carregar Dados")
uploaded_file = st.sidebar.file_uploader("Faça o upload de uma planilha (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Carregar o arquivo e inferir o tipo das colunas
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    # Detectar e converter colunas de data
    colunas_data = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    for coluna in colunas_data:
        df[coluna] = pd.to_datetime(df[coluna])
        df[f"{coluna}_dia"] = df[coluna].dt.day
        df[f"{coluna}_mes"] = df[coluna].dt.month
        df[f"{coluna}_ano"] = df[coluna].dt.year

    # Separar colunas por tipo
    colunas_categoricas = [col for col in df.columns if df[col].dtype == 'object'] + \
                          [f"{col}_dia" for col in colunas_data] + \
                          [f"{col}_mes" for col in colunas_data] + \
                          [f"{col}_ano" for col in colunas_data]
    colunas_numericas = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]

    # Seção de filtros
    st.sidebar.header("Filtros")
    colunas_disponiveis_filtro = st.sidebar.multiselect("Escolha as colunas para aplicar filtros",
                                                        colunas_categoricas + colunas_data)

    filtros = {}
    for coluna in colunas_disponiveis_filtro:
        valores_unicos = df[coluna].dropna().unique()
        selecionados = st.sidebar.multiselect(f"Filtrar {coluna}", valores_unicos, default=valores_unicos)
        filtros[coluna] = selecionados
    for coluna, valores in filtros.items():
        df = df[df[coluna].isin(valores)]

    # Configuração do gráfico
    st.subheader("Configuração do Gráfico")
    col1, col2 = st.columns(2)

    with col1:
        categoria_x = st.selectbox("Escolha a coluna para o eixo X (Categorias)", colunas_categoricas)
    with col2:
        categoria_y = st.selectbox("Escolha a coluna para o eixo Y (Valores)", colunas_numericas)

    comparativo = st.radio("Deseja um gráfico comparativo?", ("Não", "Sim"))
    nomes_conjuntos = []
    valores_comparativos = []
    if comparativo == "Sim":
        num_conjuntos = st.number_input("Quantos conjuntos deseja comparar?", min_value=2, max_value=5, value=2, step=1)
        for i in range(num_conjuntos):
            col = st.selectbox(f"Escolha a coluna de valores para o Conjunto {i + 1}", colunas_numericas, key=f"conj_{i}")
            valores_comparativos.append(df.groupby(categoria_x)[col].sum().tolist())
            nome = st.text_input(f"Nome do Conjunto {i + 1}", f"Conjunto {i + 1}", key=f"nome_conj_{i}")
            nomes_conjuntos.append(nome)

    # Escolher o tipo de gráfico
    tipo_grafico = st.selectbox("Escolha o tipo de gráfico", ["Barras", "Coluna", "Pizza", "Top10", "Linha"])
    titulo = st.text_input("Título do gráfico:", "Gráfico Personalizado")

    # Agrupamento de dados
    if tipo_grafico in ["Barras", "Coluna", "Top10", "Linha"]:
        df_agrupado = df.groupby(categoria_x)[categoria_y].sum().reset_index()
        valores = df_agrupado[categoria_y].tolist()
        categorias = df_agrupado[categoria_x].tolist()

    elif tipo_grafico == "Pizza":
        df_agrupado = df.groupby(categoria_x)[categoria_y].sum().reset_index()
        valores = df_agrupado[categoria_y].tolist()
        categorias = df_agrupado[categoria_x].tolist()

    # Gerar o gráfico
    buf = gerar_grafico(tipo_grafico, categorias, valores, titulo,
                        comparativo == "Sim", valores_comparativos, nomes_conjuntos, eixo_y=categoria_y)

    # Mostrar o gráfico
    st.image(buf, use_column_width=True)
    st.download_button("Baixar Gráfico", buf, file_name="grafico.png", mime="image/png")