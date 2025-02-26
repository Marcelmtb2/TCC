# import itertools

# # Parâmetros do circuito
# Vcc = 3.3         # Tensão de alimentação em volts
# R_up = 10000.0    # Resistor de 10 kΩ (em ohms)

# # Resistores associados a cada botão (em ohms)
# resistores_botoes = {
#     "Botão 1": 27000.0,   # 27 kΩ
#     "Botão 2": 15000.0,   # 15 kΩ
#     "Botão 3": 3300.0,    # 3,3 kΩ
#     "Botão 4": 1300.0,    # 1,3 kΩ
# }

# def resistor_paralelo(lista_resistores):
#     """Calcula o resistor equivalente para um conjunto de resistores em paralelo."""
#     if not lista_resistores:
#         return None  # Não há resistor se nenhum botão for pressionado.
#     soma_inversos = sum(1.0 / r for r in lista_resistores)
#     return 1.0 / soma_inversos

# # Lista dos botões
# botoes = list(resistores_botoes.keys())

# print("Combinações de botões pressionados e tensão no nó:")

# # Loop por todas as combinações (0 a 4 botões pressionados)
# for r in range(0, len(botoes) + 1):
#     for combinacao in itertools.combinations(botoes, r):
#         if combinacao:  # Pelo menos um botão pressionado
#             valores = [resistores_botoes[botao] for botao in combinacao]
#             R_down = resistor_paralelo(valores)
#             V_no = Vcc * R_down / (R_up + R_down)
#         else:
#             # Nenhum botão pressionado: sem caminho para o terra; assumimos V_no = Vcc.
#             V_no = Vcc
        
#         botoes_pressionados = " + ".join(combinacao) if combinacao else "Nenhum"
#         print(f"{botoes_pressionados}: V_no = {V_no:.3f} V")

import itertools

# Parâmetros do circuito
Vcc = 3.3           # Tensão de alimentação do circuito (3.3 V)
R_up = 10000.0      # Resistor superior (10 kΩ)

# Resistores associados a cada botão (em ohms)
resistores_botoes = {
    "Botão 1": 27000.0,  # 27 kΩ
    "Botão 2": 15000.0,  # 15 kΩ
    "Botão 3": 3300.0,   # 3,3 kΩ
    "Botão 4": 1300.0,   # 1,3 kΩ
}

# Configuração do ADC do ESP32 com atenuação de 11 dB:
# - Resolução: 12 bits (valores de 0 a 4095)
# - Faixa efetiva de tensão: aproximadamente 0 a 3.9 V.
adc_resolution = 4095  # valor máximo digital
V_fs_adc = 3.9         # tensão full-scale efetiva no ADC (em V)

# Calculando o LSB: menor variação de tensão que o ADC consegue detectar
LSB = V_fs_adc / 4096  # aproximadamente 0.95 mV

def resistor_paralelo(lista_resistores):
    """Calcula o resistor equivalente para resistores em paralelo."""
    if not lista_resistores:
        return None  # Nenhum resistor se nenhum botão for pressionado
    soma_inversos = sum(1.0 / r for r in lista_resistores)
    return 1.0 / soma_inversos

# Lista de botões
botoes = list(resistores_botoes.keys())

# Lista para armazenar os resultados: (combinação, tensão no nó, valor ADC)
resultados = []

# Itera por todas as combinações (0 a 4 botões pressionados)
for r in range(0, len(botoes) + 1):
    for combinacao in itertools.combinations(botoes, r):
        if combinacao:  # Caso haja pelo menos um botão pressionado
            valores = [resistores_botoes[botao] for botao in combinacao]
            R_down = resistor_paralelo(valores)
            V_no = Vcc * R_down / (R_up + R_down)
        else:
            # Nenhum botão pressionado: sem caminho para o terra (assume-se V_no = Vcc)
            V_no = Vcc
        
        # Calcula o valor digital do ADC com base na tensão medida
        # Proporção: V_no / V_fs_adc, mapeada para 0-4095
        adc_val = round(V_no / V_fs_adc * adc_resolution)
        
        resultados.append((combinacao, V_no, adc_val))

# Ordena os resultados de forma decrescente, considerando a tensão no nó
resultados.sort(key=lambda item: item[1], reverse=True)

# Exibe o LSB e os resultados ordenados
print(f"LSB (menor incremento de tensão): {LSB*1000:.2f} mV\n")
print("Combinações de botões pressionados, tensão no nó e valor ADC:")
for combinacao, V_no, adc_val in resultados:
    botoes_pressionados = " + ".join(combinacao) if combinacao else "Nenhum"
    print(f"{botoes_pressionados:20} -> V_no = {V_no:.3f} V, ADC = {adc_val}")
