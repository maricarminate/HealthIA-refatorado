"""
Script de teste rápido da API HealthIA

EXPLICAÇÃO:
Este script faz testes básicos nos endpoints da API.
Útil para verificar se tudo está funcionando depois de configurar.

COMO USAR:
1. Certifique-se de que o servidor está rodando (uvicorn app.main:app --reload)
2. Execute: python test_api.py
"""

import requests
import json

# URL base da API
BASE_URL = "http://localhost:8000/api/v1"


def test_root():
    """Testa endpoint raiz"""
    print("\n" + "="*50)
    print("🧪 Testando GET /")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200, "Endpoint raiz falhou!"
    print("✅ Teste passou!")


def test_health():
    """Testa health check"""
    print("\n" + "="*50)
    print("🧪 Testando GET /health")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200, "Health check falhou!"
    assert response.json()["status"] == "healthy", "API não está saudável!"
    print("✅ Teste passou!")


def test_diseases():
    """Testa listagem de doenças"""
    print("\n" + "="*50)
    print("🧪 Testando GET /diseases")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/diseases")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total de doenças: {data['total_diseases']}")
    print(f"Primeiras 5 doenças: {data['diseases'][:5]}")
    
    assert response.status_code == 200, "Listagem de doenças falhou!"
    assert data['total_diseases'] > 0, "Nenhuma doença encontrada!"
    print("✅ Teste passou!")


def test_predict():
    """Testa predição de diagnóstico"""
    print("\n" + "="*50)
    print("🧪 Testando POST /predict")
    print("="*50)
    
    # Casos de teste
    test_cases = [
        {
            "symptoms": "febre alta, dor no corpo, cansaço extremo",
            "expected_disease": "Febre Maculosa"  # Pode variar
        },
        {
            "symptoms": "sede constante, urinar muito, emagrecimento rápido",
            "expected_disease": "Diabetes Tipo 1"
        },
        {
            "symptoms": "tremores nas mãos, rigidez muscular, movimentos lentos",
            "expected_disease": "Doença de Parkinson"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Caso de Teste {i} ---")
        print(f"Sintomas: {test_case['symptoms']}")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"symptoms": test_case['symptoms']}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Diagnóstico: {data['diagnosis']}")
            print(f"Confiança: {data['confidence']}%")
            print(f"Sintomas processados: {data['symptoms_received'][:5]}...")
            
            assert response.status_code == 200, "Predição falhou!"
            assert "diagnosis" in data, "Diagnóstico não retornado!"
            assert data["confidence"] > 0, "Confiança inválida!"
            print("✅ Teste passou!")
        else:
            print(f"❌ Erro: {response.json()}")


def test_invalid_symptoms():
    """Testa com sintomas inválidos"""
    print("\n" + "="*50)
    print("🧪 Testando POST /predict com sintomas inválidos")
    print("="*50)
    
    # Sintomas muito curtos
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"symptoms": "a"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 422, "Deveria rejeitar sintomas inválidos!"
    print("✅ Teste passou! (Validação funcionou)")


def main():
    """Executa todos os testes"""
    print("\n" + "="*70)
    print("🏥 TESTANDO HEALTHIA API")
    print("="*70)
    print("Certifique-se de que o servidor está rodando em http://localhost:8000")
    print("="*70)
    
    try:
        # Testes básicos
        test_root()
        test_health()
        test_diseases()
        
        # Testes de predição
        test_predict()
        test_invalid_symptoms()
        
        print("\n" + "="*70)
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERRO: Não foi possível conectar à API.")
        print("Certifique-se de que o servidor está rodando:")
        print("  uvicorn app.main:app --reload")
        
    except AssertionError as e:
        print(f"\n❌ ERRO: Teste falhou - {str(e)}")
        
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {str(e)}")


if __name__ == "__main__":
    main()