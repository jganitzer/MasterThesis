# Master Thesis

# Deep Learning for Advancing Animal Breeding: A Study on Austrian Fleckvieh Cattle

This repository contains code developed for the research project titled "Deep Learning for Advancing Animal Breeding: A Study on Austrian Fleckvieh Cattle", conducted as part of a master's thesis at the University of Applied Sciences Salzburg - 
Information Technology and Systems Management.

## Abstract

This study investigates the potential of deep learning models in predicting genomic breedingvalues, utilizing genotypic and pedigree data from Austrian Fleckvieh cattle. Leveraging transformer-encoder and graph neural network-based architectures, the research aims toenhance predictive accuracy in estimating breeding values. Multiple deep learning models are explored and their performance compared against established methods like Single-Step Genomic Best Linear Unbiased Prediction (ssGBLUP) and machine learning methods such as XGBoost. Results highlight the improved predictive power of transformer-encoderbased architectures over models based on graph neural networks, ssGBLUP and XGBoost. An interpretability analysis demonstrates strong associations between single nucleotide polymorphism markers, exhibiting high attribution to model predictions and quantitative trait loci. This validates the biological significance of the markers identified by deep learning models. The study showcases the potential of transformer-encoder-based deep learning architectures as possible alternatives or extensions to existing approaches, such as genomic best unbiased predictor (GBLUP) models.

## Introduction

This repository hosts the codebase developed for the master thesis aiming to advance the state-of-the-art in genomic breeding value estimation in dairy cattle by leveraging deep learning methods for genomic prediction from genotypic data.

## Objective and Research Questions

The aim of this study is genomic evaluation of simulated data from Austrian Fleckvieh cattle for breeding value estimation, employing deep learning algorithms.
The central research question is:
- How can the application of deep learning techniques enhance the prediction of genomic breeding values in cattle using genotypic data obtained from SNP arrays?

Subsidiary questions include:
- What is the comparative predictive performance of various deep learning models against each other and baseline models in predicting breeding values from genotypic data?
- Can explainable AI methods identify and prioritize the most relevant features for the prediction of deep learning models and how well do these identified features align with the true underlying biological relevance?
- How can pedigree information be leveraged for the prediction of breeding values and what is its impact on predictive performance?

## Scope and Limitations

This study focuses on deep learning techniques based on genomic data and pedigree information. It aims to identify measurable animal characteristics without considering factors such as biological domain knowledge, environmental influences, or additional medical data. The models utilized in this study are trained using genotyped and phenotyped animals, leveraging transformer-encoder and graph neural network-based architectures. These architectures are assessed against benchmarks like Single-Step Genomic Best Linear Unbiased Prediction (ssGBLUP) and XGBoost to evaluate their performance. The evaluation of models is performed on individuals from the last generation of an animal population, aligning with the primary objective of genomic prediction.

## Conclusion

The integration of deep learning methods into genomic breeding value estimation for dairy cattle offers promising opportunities to advance animal breeding. This study utilizes transformer-encoder and graph neural network-based architectures to predict breeding values in Austrian Fleckvieh cattle using genomic and pedigree data. The results show the success of these models in predicting breeding values that closely match true breeding values. Furthermore, interpretability analysis establishes associations between markers and quantitative trait loci, highlighting correlations between key markers selected by the deep learning model and their biological relevance.
