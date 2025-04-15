with open("index_ov2640_mod.html.gz", "rb") as f:
    data = f.read()
# Formatar os dados em linhas de 16 bytes
hex_lines = []
for i in range(0, len(data), 16):
    hex_line = ', '.join(f'0x{b:02x}' for b in data[i:i+16])
    hex_lines.append(hex_line)

# Criar a string final com quebras de linha
hex_data_formatted = ',\n'.join(hex_lines)

# Salvar os dados em um arquivo
with open("saida_hex.txt", "w") as f:
    f.write(hex_data_formatted)

# Exibir a contagem de bytes
print(f"Total de bytes no arquivo: {len(data)}")
print("Arquivo 'saida_hex.txt' gerado com sucesso!")