#!/bin/bash

# Lista de nombres de secuencia
sequences=("P05_01" "P06_03" "P16_01" "P21_01") #"P09_02" "P13_03" "P16_01" "P21_01"  "P03_04" "P04_01" "P05_01" "P06_03" "P08_01" "P03_04" "P04_01" "P05_01" "P06_03" "P08_01"

# Bucle para procesar cada secuencia
for seq in "${sequences[@]}"
do
    echo "Iniciando entrenamiento para la secuencia $seq"
    sh scripts/train_clip.sh "$seq"
done