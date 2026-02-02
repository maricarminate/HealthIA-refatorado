"""
Dataset com todas as doenças e sintomas

EXPLICAÇÃO:
Este arquivo armazena todos os dados de treinamento do modelo.
Cada linha tem: (sintomas em texto, nome da doença)
O modelo aprende a reconhecer padrões entre sintomas e doenças.
"""
import pandas as pd
from typing import List, Dict


# Dataset completo com dados limpos
DATASET_DATA = [
    # Anemia Falciforme (13 exemplos)
    ("cansaço extremo dor articular severa crises dolorosas mal estar geral", "Anemia Falciforme"),
    ("cansaço extremo dor articular severa crises dolorosas", "Anemia Falciforme"),
    ("cansaço extremo dor articular episódios dolorosos dor de cabeça", "Anemia Falciforme"),
    ("cansaço severo dor articular crises de dor intensa", "Anemia Falciforme"),
    ("cansaço severo dor articular intensa dor em crises", "Anemia Falciforme"),
    ("fadiga severa dor nas juntas dor em episódios", "Anemia Falciforme"),
    ("cansaço que não passa dor articular crises álgicas", "Anemia Falciforme"),
    ("cansaço extremo dor articular severa crises dolorosas com fraqueza", "Anemia Falciforme"),
    ("fadiga severa dor nas articulações crises de dor", "Anemia Falciforme"),
    ("cansaço extremo dor articular episódios dolorosos", "Anemia Falciforme"),
    ("cansaço severo dor articular intensa dor em crises", "Anemia Falciforme"),
    ("fadiga constante dor nas juntas crises dolorosas", "Anemia Falciforme"),
    ("cansaço que não passa dor articular crises álgicas com tontura", "Anemia Falciforme"),
    
    # Artrite Reumatoide (15 exemplos)
    ("dor articular crônica juntas inchadas rigidez matinal", "Artrite Reumatoide"),
    ("articulações inflamadas dor constante rigidez", "Artrite Reumatoide"),
    ("dor nas juntas inchaço articular dificuldade de movimento com fraqueza", "Artrite Reumatoide"),
    ("dor nas juntas inchaço articular dificuldade de movimento", "Artrite Reumatoide"),
    ("dor nas articulações inchaço nas juntas rigidez matinal mal estar geral", "Artrite Reumatoide"),
    ("dor nas juntas inchaço articular rigidez ao acordar dor de cabeça", "Artrite Reumatoide"),
    ("dor nas juntas inchaço articular rigidez ao acordar", "Artrite Reumatoide"),
    ("articulações inflamadas dor constante rigidez", "Artrite Reumatoide"),
    ("articulações dolorosas inchaço persistente rigidez", "Artrite Reumatoide"),
    ("dor articular juntas inchadas dificuldade para se mover", "Artrite Reumatoide"),
    ("dor nas juntas inchaço articular rigidez ao acordar", "Artrite Reumatoide"),
    ("dor nas juntas inchaço articular dificuldade de movimento perda de apetite", "Artrite Reumatoide"),
    ("articulações rígidas e doloridas inchaço persistente", "Artrite Reumatoide"),
    ("dor nas juntas inchaço articular rigidez ao acordar perda de apetite", "Artrite Reumatoide"),
    ("dor articular juntas inchadas dificuldade para se mover com fraqueza", "Artrite Reumatoide"),
    
    # Diabetes Tipo 1 (13 exemplos)
    ("sede constante micção frequente perda de peso rápida", "Diabetes Tipo 1"),
    ("muita sede urinar muito emagrecimento", "Diabetes Tipo 1"),
    ("sede excessiva poliúria perda de peso inexplicável", "Diabetes Tipo 1"),
    ("muita sede urina frequente emagrecimento súbito", "Diabetes Tipo 1"),
    ("sede excessiva urina frequente perda de peso mal estar geral", "Diabetes Tipo 1"),
    ("sede excessiva urina frequente perda de peso", "Diabetes Tipo 1"),
    ("sede intensa micção excessiva perda de peso mal estar geral", "Diabetes Tipo 1"),
    ("sede excessiva poliúria perda de peso inexplicável", "Diabetes Tipo 1"),
    ("sede constante urinar muito emagrecimento rápido", "Diabetes Tipo 1"),
    ("sede intensa urina em excesso emagrecimento súbito", "Diabetes Tipo 1"),
    ("sede constante urinar muito emagrecimento rápido com tontura", "Diabetes Tipo 1"),
    ("muita sede frequência urinária aumentada emagrecimento", "Diabetes Tipo 1"),
    ("muita sede urina frequente emagrecimento súbito perda de apetite", "Diabetes Tipo 1"),
    
    # Doença de Alzheimer (13 exemplos)
    ("esquecimento constante confusão perda de referências com fraqueza", "Doença de Alzheimer"),
    ("esquecimento confusão perda de noção de tempo e lugar", "Doença de Alzheimer"),
    ("perda de memória progressiva confusão dificuldade de orientação", "Doença de Alzheimer"),
    ("problemas de memória confusão mental desorientação temporal", "Doença de Alzheimer"),
    ("problemas de memória confusão dificuldade de localização", "Doença de Alzheimer"),
    ("perda de memória confusão mental desorientação", "Doença de Alzheimer"),
    ("esquecimento constante confusão perda de referências", "Doença de Alzheimer"),
    ("perda de memória confusão frequente dificuldade de localização", "Doença de Alzheimer"),
    ("problemas de memória severos confusão mental desorientação", "Doença de Alzheimer"),
    ("esquecimento frequente confusão perda de orientação", "Doença de Alzheimer"),
    
    # Doença de Crohn (12 exemplos)
    ("cólicas abdominais intensas diarreia sanguinolenta", "Doença de Crohn"),
    ("dor na região abdominal diarreia frequente fraqueza", "Doença de Crohn"),
    ("cólicas fortes diarreia com muco emagrecimento perda de apetite", "Doença de Crohn"),
    ("dor na região abdominal diarreia frequente fraqueza com tontura", "Doença de Crohn"),
    ("dor abdominal severa diarreia crônica fadiga", "Doença de Crohn"),
    ("dor no abdômen fezes líquidas perda de peso rápida", "Doença de Crohn"),
    ("cólicas abdominais intensas diarreia sanguinolenta com fraqueza", "Doença de Crohn"),
    ("cólicas fortes diarreia com muco emagrecimento", "Doença de Crohn"),
    ("cólicas abdominais diarreia persistente perda de apetite mal estar geral", "Doença de Crohn"),
    
    # Doença de Lyme (14 exemplos)
    ("febre baixa cansaço dor no corpo manchas na pele", "Doença de Lyme"),
    ("febre baixa cansaço constante dor no corpo manchas", "Doença de Lyme"),
    ("febre baixa cansaço severo dor no corpo manchas cutâneas mal estar geral", "Doença de Lyme"),
    ("febre baixa cansaço severo dor no corpo manchas cutâneas com tontura", "Doença de Lyme"),
    ("febre baixa cansaço constante dor no corpo manchas perda de apetite", "Doença de Lyme"),
    ("febre intermitente fadiga dor muscular erupção com tontura", "Doença de Lyme"),
    ("febre persistente fadiga severa dor muscular erupções", "Doença de Lyme"),
    ("febre fadiga dor muscular erupção cutânea com fraqueza", "Doença de Lyme"),
    ("febre intermitente cansaço dor corporal lesões cutâneas", "Doença de Lyme"),
    ("febre fadiga dor muscular erupção cutânea mal estar geral", "Doença de Lyme"),
    
    # Doença de Parkinson (13 exemplos)
    ("tremores nas mãos rigidez muscular movimentos lentos", "Doença de Parkinson"),
    ("tremores nas extremidades rigidez lentidão motora dor de cabeça", "Doença de Parkinson"),
    ("tremor nas mãos rigidez corporal bradicinesia", "Doença de Parkinson"),
    ("tremor nas mãos rigidez muscular instabilidade postural", "Doença de Parkinson"),
    ("tremor característico músculos tensos movimentos lentos", "Doença de Parkinson"),
    ("tremor de repouso rigidez movimentos prejudicados", "Doença de Parkinson"),
    ("tremor característico músculos tensos movimentos lentos mal estar geral", "Doença de Parkinson"),
    ("tremores rigidez muscular perda de equilíbrio", "Doença de Parkinson"),
    ("tremor em repouso rigidez dificuldade para se mover dor de cabeça", "Doença de Parkinson"),
    
    # Doença de Wilson (10 exemplos)
    ("cansaço dor na barriga tremor problemas no fígado", "Doença de Wilson"),
    ("cansaço dor na região abdominal tremor hepatopatia", "Doença de Wilson"),
    ("fadiga constante dor abdominal tremores hepatopatia", "Doença de Wilson"),
    ("cansaço severo dor na barriga tremor disfunção hepática mal estar geral", "Doença de Wilson"),
    ("cansaço severo dor na barriga tremor disfunção hepática", "Doença de Wilson"),
    ("cansaço constante dor no abdômen tremor problemas hepáticos", "Doença de Wilson"),
    ("fadiga dor abdominal tremores problemas hepáticos", "Doença de Wilson"),
    ("cansaço crônica dor abdominal tremores disfunção do fígado", "Doença de Wilson"),
    ("fadiga extrema dor abdominal tremores hepatopatia", "Doença de Wilson"),
    
    # Esclerose Lateral (11 exemplos)
    ("braços sem força fala difícil atrofia muscular", "Esclerose Lateral"),
    ("braços fracos problemas na fala contrações musculares", "Esclerose Lateral"),
    ("perda de força muscular dificuldade para engolir dor de cabeça", "Esclerose Lateral"),
    ("dificuldade para levantar os braços fala alterada", "Esclerose Lateral"),
    ("perda de força fala lenta rigidez muscular com fraqueza", "Esclerose Lateral"),
    ("perda de força nos membros fala arrastada espasmos", "Esclerose Lateral"),
    ("perda de força muscular dificuldade para engolir", "Esclerose Lateral"),
    ("perda de força nos membros fala arrastada espasmos perda de apetite", "Esclerose Lateral"),
    ("dificuldade para mover os braços fala prejudicada tremores", "Esclerose Lateral"),
    
    # Esclerose Múltipla (13 exemplos)
    ("fadiga persistente perda de força visão dupla formigamento com tontura", "Esclerose Múltipla"),
    ("fadiga crônica coordenação prejudicada visão turva", "Esclerose Múltipla"),
    ("fadiga fraqueza nas pernas visão turva formigamento", "Esclerose Múltipla"),
    ("cansaço severo fraqueza muscular visão dupla dormência", "Esclerose Múltipla"),
    ("cansaço severo fraqueza muscular problemas visuais tontura", "Esclerose Múltipla"),
    ("cansaço persistente perda de força visão dupla formigamento com tontura", "Esclerose Múltipla"),
    ("cansaço extremo pernas fracas visão embaçada formigamento", "Esclerose Múltipla"),
    ("fadiga persistente perda de força visão dupla formigamento dor de cabeça", "Esclerose Múltipla"),
    ("cansaço persistente perda de força visão dupla formigamento", "Esclerose Múltipla"),
    ("fadiga perda de equilíbrio problemas de visão dormência", "Esclerose Múltipla"),
    
    # Febre Maculosa (13 exemplos)
    ("temperatura alta dor nos músculos cansaço que não passa", "Febre Maculosa"),
    ("febre alta dor no corpo todo cansaço extremo", "Febre Maculosa"),
    ("febre persistente dor muscular generalizada erupção cutânea", "Febre Maculosa"),
    ("febre alta dor nas costas náusea vômito", "Febre Maculosa"),
    ("temperatura elevada dores musculares intensas cansaço severo com fraqueza", "Febre Maculosa"),
    ("temperatura elevada dor generalizada náusea", "Febre Maculosa"),
    ("temperatura elevada dores musculares intensas fadiga severa", "Febre Maculosa"),
    ("febre persistente fadiga extrema erupções na pele", "Febre Maculosa"),
    ("dor de cabeça intensa febre alta dor muscular", "Febre Maculosa"),
    
    # Fibromialgia (13 exemplos)
    ("dor no corpo todo cansaço constante dificuldade para dormir", "Fibromialgia"),
    ("dores musculares difusas fadiga severa sono ruim", "Fibromialgia"),
    ("dor muscular difusa cansaço severo dificuldades do sono com tontura", "Fibromialgia"),
    ("dor em múltiplos pontos fadiga crônica insônia severa", "Fibromialgia"),
    ("dor corporal generalizada cansaço persistente sono fragmentado", "Fibromialgia"),
    ("dor em vários pontos do corpo cansaço extremo insônia perda de apetite", "Fibromialgia"),
    ("dor muscular generalizada fadiga crônica insônia perda de apetite", "Fibromialgia"),
    ("dor generalizada cansaço que não melhora problemas para dormir", "Fibromialgia"),
    ("dor muscular generalizada cansaço crônica insônia", "Fibromialgia"),
    
    # Hipertireoidismo (13 exemplos)
    ("emagrecimento rápido taquicardia ansiedade constante mal estar geral", "Hipertireoidismo"),
    ("perda de peso palpitações frequentes nervosismo", "Hipertireoidismo"),
    ("emagrecimento súbito batimentos acelerados ansiedade", "Hipertireoidismo"),
    ("emagrecimento rápido taquicardia ansiedade constante dor de cabeça", "Hipertireoidismo"),
    ("emagrecimento palpitações estado de nervosismo", "Hipertireoidismo"),
    ("perda de peso inexplicável coração acelerado agitação", "Hipertireoidismo"),
    ("emagrecimento súbito coração disparado ansiedade com tontura", "Hipertireoidismo"),
    ("perda de peso rápida batimentos irregulares agitação", "Hipertireoidismo"),
    ("perda de peso palpitações frequentes nervosismo perda de apetite", "Hipertireoidismo"),
    
    # Hipotireoidismo (14 exemplos)
    ("fadiga severa obesidade frieza excessiva", "Hipotireoidismo"),
    ("cansaço extremo obesidade progressiva sensação de frio", "Hipotireoidismo"),
    ("cansaço severo aumento de peso frio excessivo com tontura", "Hipotireoidismo"),
    ("cansaço crônica aumento de peso inexplicável intolerância ao frio", "Hipotireoidismo"),
    ("cansaço extremo ganho de peso inexplicável sensibilidade ao frio", "Hipotireoidismo"),
    ("cansaço que não passa ganho de peso rápido frio constante dor de cabeça", "Hipotireoidismo"),
    ("fadiga constante obesidade intolerância ao frio", "Hipotireoidismo"),
    ("fadiga crônica aumento de peso inexplicável intolerância ao frio com tontura", "Hipotireoidismo"),
    ("cansaço constante aumento de peso sensibilidade ao frio", "Hipotireoidismo"),
    
    # Lúpus (13 exemplos)
    ("dor articular fadiga crônica sensibilidade ao sol com fraqueza", "Lúpus"),
    ("juntas doloridas cansaço que não passa manchas na face", "Lúpus"),
    ("dor articular fadiga crônica sensibilidade ao sol", "Lúpus"),
    ("dor nas articulações cansaço extremo problemas de pele", "Lúpus"),
    ("dor nas juntas fadiga que não melhora erupção facial", "Lúpus"),
    ("fadiga constante dor nas articulações sensibilidade à luz", "Lúpus"),
    ("dor articular generalizada fadiga severa erupções cutâneas", "Lúpus"),
    ("articulações rígidas cansaço constante fotossensibilidade", "Lúpus"),
    ("cansaço extremo articulações inchadas fotofobia", "Lúpus"),
    
    # Miastenia Gravis (13 exemplos)
    ("perda de força cansaço diplopia problemas de deglutição", "Miastenia Gravis"),
    ("perda de força cansaço extremo diplopia problemas para engolir", "Miastenia Gravis"),
    ("fraqueza muscular fadiga severa visão dupla dificuldade para engolir", "Miastenia Gravis"),
    ("perda de força cansaço diplopia problemas para engolir", "Miastenia Gravis"),
    ("perda de força muscular cansaço problemas visuais dificuldade de deglutição perda de apetite", "Miastenia Gravis"),
    ("fraqueza nos músculos fadiga severa visão dupla disfagia com fraqueza", "Miastenia Gravis"),
    ("fraqueza nos músculos fadiga constante visão dupla dificuldade de deglutição", "Miastenia Gravis"),
    ("fraqueza muscular fadiga visão dupla dificuldade para engolir com tontura", "Miastenia Gravis"),
    
    # Porfiria (12 exemplos)
    ("dor intensa na barriga enjoo vômito confusão perda de apetite", "Porfiria"),
    ("dor abdominal intensa náusea persistente vômito confusão mental com fraqueza", "Porfiria"),
    ("dor abdominal severa enjoo vômito confusão mental", "Porfiria"),
    ("dor abdominal severa náusea vômito confusão mental com tontura", "Porfiria"),
    ("dor severa na barriga enjoo vômito persistente confusão", "Porfiria"),
    ("dor severa na barriga enjoo vômito frequente alterações cognitivas", "Porfiria"),
    ("dor forte no abdômen enjoo constante vômito confusão", "Porfiria"),
    ("dor intensa no abdômen enjoo severo vômito confusão perda de apetite", "Porfiria"),
    ("dor abdominal aguda enjoo vômito confusão mental mal estar geral", "Porfiria"),
    ("dor abdominal forte náusea constante vômito alterações mentais", "Porfiria"),
    
    # Sarcoidose (14 exemplos)
    ("cansaço severo falta de ar aos esforços tosse febre", "Sarcoidose"),
    ("cansaço crônica falta de ar tosse seca persistente febre mal estar geral", "Sarcoidose"),
    ("cansaço extremo falta de ar tosse persistente febre baixa", "Sarcoidose"),
    ("cansaço dispneia aos esforços tosse febre baixa", "Sarcoidose"),
    ("fadiga constante dispneia tosse seca febre intermitente dor de cabeça", "Sarcoidose"),
    ("fadiga falta de ar tosse seca febre baixa", "Sarcoidose"),
    ("cansaço constante dificuldade para respirar tosse febre intermitente", "Sarcoidose"),
    ("fadiga severa dificuldade respiratória tosse seca febre mal estar geral", "Sarcoidose"),
    ("cansaço dificuldade para respirar tosse febre", "Sarcoidose"),
    
    # Síndrome da Fadiga Crônica (13 exemplos)
    ("fadiga severa crônica dor no corpo todo problemas de foco", "Síndrome da Fadiga Crônica"),
    ("cansaço que não alivia dor muscular difusa névoa cerebral", "Síndrome da Fadiga Crônica"),
    ("fadiga extrema constante dor generalizada dificuldade mental", "Síndrome da Fadiga Crônica"),
    ("fadiga severa crônica dor no corpo todo problemas de foco dor de cabeça", "Síndrome da Fadiga Crônica"),
    ("cansaço constante e severo dores musculares problemas cognitivos", "Síndrome da Fadiga Crônica"),
    ("cansaço severo crônica dor no corpo todo problemas de foco", "Síndrome da Fadiga Crônica"),
    ("cansaço extremo que não melhora dor muscular problemas de memória perda de apetite", "Síndrome da Fadiga Crônica"),
    ("fadiga extrema dor generalizada dificuldade de concentração", "Síndrome da Fadiga Crônica"),
    ("cansaço extremo e persistente dor muscular confusão", "Síndrome da Fadiga Crônica"),
    
    # Síndrome de Sjögren (11 exemplos)
    ("secura na boca olhos ressecados cansaço dor articular crônica com fraqueza", "Síndrome de Sjögren"),
    ("boca seca olhos secos severos fadiga dor nas articulações", "Síndrome de Sjögren"),
    ("secura oral olhos ressecados cansaço constante dor articular dor de cabeça", "Síndrome de Sjögren"),
    ("secura oral olhos ressecados cansaço constante dor articular", "Síndrome de Sjögren"),
    ("boca ressecada olhos secos crônicos fadiga dor articular dor de cabeça", "Síndrome de Sjögren"),
    ("boca ressecada olhos secos fadiga constante dor articular", "Síndrome de Sjögren"),
    ("boca seca persistente olhos secos fadiga dor articular", "Síndrome de Sjögren"),
    ("secura na boca olhos ressecados cansaço extremo dor nas juntas", "Síndrome de Sjögren"),
    ("boca seca olhos secos fadiga dor articular perda de apetite", "Síndrome de Sjögren"),
]


def get_dataset_dataframe() -> pd.DataFrame:
    """
    Retorna o dataset completo como DataFrame do pandas.
    
    EXPLICAÇÃO:
    Converte nossa lista de tuplas em um DataFrame (tabela)
    com duas colunas: 'sintomas' e 'diagnostico'
    
    Returns:
        DataFrame com colunas 'sintomas' e 'diagnostico'
    """
    df = pd.DataFrame(DATASET_DATA, columns=["sintomas", "diagnostico"])
    return df


def get_available_diseases() -> List[str]:
    """
    Retorna lista única de todas as doenças no dataset.
    
    EXPLICAÇÃO:
    Pega todas as doenças únicas que o modelo pode diagnosticar.
    Útil para mostrar ao usuário quais doenças são suportadas.
    
    Returns:
        Lista de nomes de doenças
    """
    df = get_dataset_dataframe()
    diseases = sorted(df["diagnostico"].unique().tolist())
    return diseases


def get_disease_info() -> Dict[str, int]:
    """
    Retorna informações sobre o dataset.
    
    EXPLICAÇÃO:
    Estatísticas úteis: quantas doenças, quantos exemplos, etc.
    
    Returns:
        Dicionário com informações do dataset
    """
    df = get_dataset_dataframe()
    return {
        "total_samples": len(df),
        "total_diseases": df["diagnostico"].nunique(),
        "diseases": get_available_diseases()
    }