# td_prediction_llm
This repository contains the source code and experiment outputs associated with our MOC2025 workshop paper at the DECLARE conference.

The paper presents a novel LLM/AI-enabled workflow for automatic labeling, combined with XAI human-in-the-loop quality control. We curated this workflow from a case study on detecting technical debt, aiming to support software project management in making informed decisions.

This work extends a thesis study that used a primary LLM as the labeling judge alongside classical ML methods to predict technical debt but suffered from feature leakage, shortcut learning, and challenges in handling imbalanced data. Key contributions include a curated workflow design, improved prompt engineering, and practical lessons learned to avoid shortcut learning or feature leakage when using LLM-generated labels. We also evaluate performance on imbalanced datasets. 

![workflowdiagram drawio](workflowdiagram.drawio.svg)


