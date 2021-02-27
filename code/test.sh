#!/bin/bash


echo "==================== Test 1 ===================="

./mlp 0.001 100 50,60,50,10 ../data/heart_disease_norm.csv 1
./mlp 0.001 100 50,60,50,10 ../data/heart_disease_norm.csv 2
./mlp 0.001 100 50,60,50,10 ../data/heart_disease_norm.csv 4


echo "==================== Test 2 ===================="

./mlp 0.001 10 200,500 ../data/heart_disease_norm.csv 1
./mlp 0.001 10 200,500 ../data/heart_disease_norm.csv 2
./mlp 0.001 10 200,500 ../data/heart_disease_norm.csv 4


echo "==================== Test 3 ===================="

./mlp 0.001 10 100,2000 ../data/heart_disease_norm.csv 1
./mlp 0.001 10 100,2000 ../data/heart_disease_norm.csv 2
./mlp 0.001 10 100,2000 ../data/heart_disease_norm.csv 4


echo "==================== Test 4 ===================="

./mlp 0.001 10 90,70,50,20,10 ../data/heart_disease_norm.csv 1
./mlp 0.001 10 90,70,50,20,10 ../data/heart_disease_norm.csv 2
./mlp 0.001 10 90,70,50,20,10 ../data/heart_disease_norm.csv 4
