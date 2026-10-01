def ReLu(suma):
    return max(0, suma)

def mlp_con_pesos(pesos1, pesos2, pesos_finales, entrada, sesgo1, sesgo2, sesgo_final):
    tasa_aprendizaje = 0.1
    epocas = 1500

    for epoca in range(epocas):
        for tupla, resultado in entrada:
            suma1 = pesos1[0] * tupla[0] + pesos1[1] * tupla[1] + sesgo1
            relu1 = ReLu(suma1)

            suma2 = pesos2[0] * tupla[0] + pesos2[1] * tupla[1] + sesgo2
            relu2 = ReLu(suma2)

            suma_final = pesos_finales[0] * relu1 + pesos_finales[1] * relu2 + sesgo_final
            resultado_final = ReLu(suma_final)
            
            error = resultado - resultado_final

            error_neurona1 = error * pesos_finales[0]
            error_neurona2 = error * pesos_finales[1]

            pesos_finales[0] += error * tasa_aprendizaje * relu1
            pesos_finales[1] += error * tasa_aprendizaje * relu2
            sesgo_final      += tasa_aprendizaje * error

            pesos1[0] += tasa_aprendizaje * error_neurona1 * tupla[0]
            pesos1[1] += tasa_aprendizaje * error_neurona1 * tupla[1]
            sesgo1    += tasa_aprendizaje * error_neurona1

            pesos2[0] += tasa_aprendizaje * error_neurona2 * tupla[0]
            pesos2[1] += tasa_aprendizaje * error_neurona2 * tupla[1]
            sesgo2    += tasa_aprendizaje * error_neurona2

        if (epoca + 1) % 200 == 0:
            print(f"\n--- ÉPOCA {epoca + 1} ---")
            print(f"  [Capa Oculta 1] Pesos: {[round(p, 4) for p in pesos1]} | Sesgo: {round(sesgo1, 4)}")
            print(f"  [Capa Oculta 2] Pesos: {[round(p, 4) for p in pesos2]} | Sesgo: {round(sesgo2, 4)}")
            print(f"  [Capa Final]    Pesos: {[round(p, 4) for p in pesos_finales]} | Sesgo: {round(sesgo_final, 4)}")
            print("  --- Predicciones ---")
            for t, res_esp in entrada:
                s1 = ReLu(pesos1[0] * t[0] + pesos1[1] * t[1] + sesgo1)
                s2 = ReLu(pesos2[0] * t[0] + pesos2[1] * t[1] + sesgo2)
                r_fin = ReLu(pesos_finales[0] * s1 + pesos_finales[1] * s2 + sesgo_final)
                print(f"    Entrada: {t} | Esperado: {res_esp} | Predicción: {round(r_fin, 4)}")

datos_xor = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

pesos_1 = [0.5, 0.5]
pesos_2 = [0.5, 0.5]
pesos_f = [1.0, 1.0]
sesgo_1 = -0.2
sesgo_2 = -0.2
sesgo_f = -0.5

print("Iniciando entrenamiento y monitoreo de pesos...")
mlp_con_pesos(pesos_1, pesos_2, pesos_f, datos_xor, sesgo_1, sesgo_2, sesgo_f)