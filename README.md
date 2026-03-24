# Topological Analysis of Rumor Propagation (PHEME)

This repository contains code for analyzing the structure of rumor and non-rumor conversations in social media using Topological Data Analysis (TDA) and the PHEME dataset.

## Overview

The goal of this project is to study whether rumors and non-rumors exhibit different structural patterns in how they spread through conversation threads and evaluate their statistical significance.

## Dataset

We use the PHEME dataset, available at https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619?file=6453753, 
which contains conversation threads from Twitter labeled as
* rumour
* non-rumour

Each thread consists of a source tweet and a set of replies (tree structure).

## Pipeline

The main pipeline consists of the following steps:

1. Preprocessing
	* Parse raw PHEME threads into a tabular format
	* Extract relevant fields:
	* thread_id
	* tweet_id
	* user_id
	* created_at_ts
	* in_reply_to_status_id
	* label, event

2. Graph construction
WIP

[...]

## Requirements

pip install -r requirements.txt


