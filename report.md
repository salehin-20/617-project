# 617 Project Report

## 1. Overview
This project focuses on collecting NYISO load data and weather data, merging them, and preparing a dataset for analysis.

## 2. Data Sources
- NYISO Load Data (EIA API / MIS fallback)
- Open-Meteo Weather API

## 3. Methodology
1. Fetch load data  
2. Fetch weather data  
3. Clean and preprocess  
4. Merge datasets  
5. Feature engineering  

## 4. Features Added
- Holiday indicator  
- Extreme temperature flag  

## 5. Output
Final dataset is saved as a Parquet file for further modeling.

