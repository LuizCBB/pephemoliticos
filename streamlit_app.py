
import streamlit as st
st.set_page_config(layout="wide")

from tqdm import tqdm
import pickle
import screed
from scipy.special import softmax
import pandas as pd
import mmh3
import time


alphabet = ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 
            'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C']


def change_aa(peptide, index, aa):
    return peptide[:index] + aa + peptide[index+1:]


def mutant_peptides(peptide):
    mutants = []
    for aa in alphabet:
        for i in range(len(peptide)):        
            if aa != peptide[i]:
                mutants.append( change_aa(peptide, i, aa) )
                
    return mutants 


def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)

    intersection = len(a.intersection(b))
    n_a = len(a)
    n_b = len(b)

    return intersection / (n_a + n_b - intersection)

def clear_sequence(sequence):
    sequence = sequence.upper()
    clear_seq = ""
    for i in sequence:
        if i in alphabet:
            clear_seq += i
    return clear_seq

def build_kmers(sequence):
    sequence = clear_sequence(sequence)
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers

def step1(query_sequences):
    progress_text = "Processando as sequências de entrada em k-mers (k = " + str(ksize) + ") ... "
    print(progress_text)
    
    kmers_list_query = []
    counter = 0
    for i in tqdm(query_sequences):
        counter += 1
        my_bar1.progress(round( (counter/len(query_sequences))*100 ), text=progress_text + str(counter) + " of " + str(len(query_sequences)))
        kmers_list_query.append( build_kmers(i) )
        
    return kmers_list_query

def step2(kmers_list_query, kmers_list_db):
    progress_text = "Calculando o índice de similaridade de Jaccard entre pares de sequências usando seus respectivos dados kmer ..."
    print(progress_text)
    
    matrix_query = []
    counter = 0
    for i in tqdm(range(len(kmers_list_query))):
        counter += 1
        my_bar2.progress(round( (counter/len(kmers_list_query))*100 ), text=progress_text + str(counter) + " of " + str(len(kmers_list_query)))
        js = []
        for j in range(len(kmers_list_db)):
            js.append( jaccard_similarity(kmers_list_query[i], kmers_list_db[j]) * 100)
        matrix_query.append(js)

    return matrix_query



#########################################################################################################################################


labels = ['Não hemolítico', 'Hemolítico'] 
 
st.markdown("""
    <style>
        button {
            height: auto;
            width: 100% !important;
            
        }
        p {
            text-align: justify
        }
    </style>
""", unsafe_allow_html=True) 
 
st.header('Predição e otimização de peptídeos antibacterianos não hemolíticos')
st.markdown("""<p>Na aba "Classificador" é possível avaliar se peptídeos antibacterianos tem potencial ou não de apresentar atividade hemolítica. A ferramenta
 pode receber uma ou mais sequências peptídicas em formato FASTA. </p>
 <p>Na aba "Peptídeos mutantes" a ferramenta pode ser usada para prever a atividade hemolítica de um único peptídeo e para gerar seus análogos com 
 sucessivas substituições de aminoácidos em cada posição. Esse recurso ajuda o usuário a selecionar os peptídeos mutantes, em relação ao peptídeo original, que podem
 ter uma probabilidade mais alta de não apresentarem atividade hemolítica.</p>
 """, unsafe_allow_html=True) 


# Criando guias
guias = st.tabs(["Classificador", "Peptídeos mutantes"])

# Conteúdo das guias
with guias[0]:
    if "sequences" not in st.session_state:
        st.session_state["sequences"] = ""

    sequences_area = st.text_area("Cole sua sequência em formato FASTA ou use o exemplo", value = st.session_state["sequences"], height = 300)
    
    option = st.selectbox(
    "Selecionar um modelo preditivo:",
    ("Modelo 1", "Modelo 2"),
    )
    
    if option == "Modelo 1":
        with open("modelo_LR_HemoPI-1_k3_treino.pkl", 'rb') as f:
            model = pickle.load(f)
        f.close()
        
        with open("matriz_dataset_treino_HemoPI-1_k3_kmers_list.pkl", "rb") as input_file:
            kmers_list_db = pickle.load(input_file)    
        input_file.close()
        
        ksize = 3
         
    elif option == "Modelo 2":
        with open("modelo_LR_HAPPENN_k5_treino.pkl", 'rb') as f:
            model = pickle.load(f)
        f.close()
        
        with open("matriz_dataset_treino_HAPPENN_k5_kmers_list.pkl", "rb") as input_file:
            kmers_list_db = pickle.load(input_file)    
        input_file.close()
        
        ksize = 5
        
    br = st.button("Executar", type="primary")
    ex = st.button("Use um exemplo")
    cl = st.button("Limpar")

    query_sequences = []
    query_labels = [] 
    probabilidades = []   
    
    if br:
        temp = open("temp.fas", "w")
        temp.write(sequences_area)
        temp.close()
        
        for record in screed.open("temp.fas"):
            name = record.name
            sequence = record.sequence
            
            
            query_labels.append(name)
            query_sequences.append(sequence)
            
        n_queries = len(query_sequences)
        

        
        my_bar1 = st.progress(0, text="")
        kmers_list_query = step1(query_sequences)

        my_bar2 = st.progress(0, text="")
        matrix = step2(kmers_list_query, kmers_list_db)
        
        predicted_class = []
        query_name = []
                
        counter_label = 0
        counter = n_queries
        while (counter > 0):
                       
            query = matrix[-(counter)]                        
                
            yhat = model.predict([query])
            prob = softmax( model.predict_proba([query])[0] )*100              
               
            predicted_class.append(labels[yhat[0]]) 
            query_name.append(query_labels[counter_label]) 
            probabilidades.append(max(prob))
                      
            counter -= 1
            counter_label += 1 
                
        d = {'Nome do peptídeo de consulta': query_name, 'Classe predita': predicted_class, 'Probabilidade' : probabilidades}
        df = pd.DataFrame(data=d,index=None)
                
        st.table(df)
        

     
    example = """>peptide_pm_1 (hemolytic)
GIFGKILGVGKKVLCGLSGVC
>peptide_pm_3 (hemolytic)
KFFKFFKFF
>lcl|2596 (non-hemolytic)
CVHWQTNTARTSCIGP
>lcl|4338 (non-hemolytic)
GIHDILKYGKPA

    """


    if ex:
        st.session_state["sequences"] = example
        st.rerun()  


    if cl:
        st.session_state["sequences"] = ""
        st.rerun() 
        
with guias[1]:
    if "sequence_mutant" not in st.session_state:
        st.session_state["sequence_mutant"] = ""

    sequences_area_m = st.text_area("Cole sua sequência em formato FASTA ou use o exemplo", value = st.session_state["sequence_mutant"], height = 100)
    
    
    option2 = st.selectbox(
    "Selecionar o modelo preditivo:",
    ("Modelo 1", "Modelo 2"),
    )
    
    if option2 == "Modelo 1":
        with open("modelo_LR_HemoPI-1_k3_treino.pkl", 'rb') as f:
            model = pickle.load(f)
        f.close()
        
        with open("matriz_dataset_treino_HemoPI-1_k3_kmers_list.pkl", "rb") as input_file:
            kmers_list_db = pickle.load(input_file)    
        input_file.close()
        
        ksize = 3
         
    elif option2 == "Modelo 2":
        with open("modelo_LR_HAPPENN_k5_treino.pkl", 'rb') as f:
            model = pickle.load(f)
        f.close()
        
        with open("matriz_dataset_treino_HAPPENN_k5_kmers_list.pkl", "rb") as input_file:
            kmers_list_db = pickle.load(input_file)    
        input_file.close()
        
        ksize = 5
        
    br_m = st.button("Criar peptídeos mutantes e realizar a classificação", type="primary")
    ex_m = st.button("Usar um exemplo")
    cl_m = st.button("Limpar o formulário")

    query_sequences = []
    query_labels = [] 
    probabilidades = []   
    
    if br_m:
        temp_m = open("temp_m.fas", "w")
        temp_m.write(sequences_area_m.strip())
        temp_m.close()
        
        for record in screed.open("temp_m.fas"):
            name = record.name
            sequence = record.sequence
            break
            
        query_sequences.append(sequence)
        query_sequences = query_sequences + mutant_peptides(sequence)            
               
        n_queries = len(query_sequences)
        
        query_labels.append(name)
        for i in range(n_queries):
            query_labels.append(name+"_mutant_"+str(i+1))       

        
        my_bar1 = st.progress(0, text="")
        kmers_list_query = step1(query_sequences)

        my_bar2 = st.progress(0, text="")
        matrix = step2(kmers_list_query, kmers_list_db)
        
        predicted_class = []
        query_name = []
        query_mutant = []        
        counter_label = 0
        counter = n_queries
        while (counter > 0):
                       
            query = matrix[-(counter)]                        
                
            yhat = model.predict([query])
            prob = softmax( model.predict_proba([query])[0] )*100
            query_mutant.append(query_sequences[counter_label])            
               
            predicted_class.append(labels[yhat[0]]) 
            query_name.append(query_labels[counter_label]) 
            probabilidades.append(max(prob))
                      
            counter -= 1
            counter_label += 1 
                
        d = {'Nome do peptídeo de consulta': query_name, 'Sequência peptídica': query_mutant, 'Classe predita': predicted_class, 'Probabilidade' : probabilidades}
        df = pd.DataFrame(data=d,index=None)
                
        #st.table(df)
        with st.expander(":blue[**Melhores resultados**]"):
            df_pep_selvagem = df.iloc[0]
            st.write("Dados do peptídeo original (selvagem):")
            st.write("Nome: ", df_pep_selvagem["Nome do peptídeo de consulta"]) 
            st.write("Sequência peptídica: ", df_pep_selvagem["Sequência peptídica"])
            st.write("Classe predita: ", df_pep_selvagem["Classe predita"])
            st.write("Probabilidade: ", str(df_pep_selvagem["Probabilidade"]))
            
            
            st.write("\n\n")
            st.write("Melhores sequências mutantes encontradas:")
            result = df.loc[ df['Probabilidade'] == max(probabilidades)].loc[ df['Classe predita'] == "Não hemolítico"]
            st.write(result)
        
        with st.expander(":blue[**Resultados da varredurra completa**]"):
            st.table(df)

    example_m = """>lcl|4736(non-hemolytic)
DLWNSIKDMAAAAGRAALNAVTGMVNQ
    """


    if ex_m:
        st.session_state["sequence_mutant"] = example_m
        st.rerun()  


    if cl_m:
        st.session_state["sequence_mutant"] = ""
        st.rerun()    
    
    

    
    
    
    
    
    
    
    
    
    
    
 
    




    
