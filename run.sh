#!/bin/bash

# ⚠️ Before starting the experiments, make sure the following are properly set:
# 1. Deploy and export your OpenAI API key (e.g., via OPENAI_API_KEY environment variable).
# 2. Ensure the embedding model is available and ready to use.
# 3. The list "[5,10,20,30,50,70,100]" specifies different top-k values for retrieval.
# 4. The list "[0]" specifies noise levels (0 means no noise).
# 5. Replace "date_here" with the actual date or any custom tag to help with experiment tracking and result organization.

# Step 1: Build datasets
python ./codes/datasets/make_datasets.py musique    # Build the Musique dataset
python ./codes/datasets/make_datasets.py twowiki    # Build the TwoWiki dataset
python ./codes/datasets/make_datasets.py webq       # Build the WebQuestions dataset

# Step 2: Run evaluation scripts with multiple top-k settings
python ./codes/eval_scripts/run_all.py date_here full ChatGPT_request "[5,10,20,30,50,70,100]" "[0]" final dynamic musique  # Evaluate Musique dataset
python ./codes/eval_scripts/run_all.py date_here full ChatGPT_request "[5,10,20,30,50,70,100]" "[0]" final dynamic twowiki  # Evaluate TwoWiki dataset
python ./codes/eval_scripts/run_all.py date_here full ChatGPT_request "[5,10,20,30,50,70,100]" "[0]" final dynamic webq     # Evaluate WebQ dataset
