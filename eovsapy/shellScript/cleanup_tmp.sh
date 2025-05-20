#!/bin/bash

# Set base directory
BASE_DIR="/common/webplots/lwa-data/tmp"

# Find and delete .tar.gz files under data-request/ older than 24 hours
find "$BASE_DIR"/data-request -type f -name "*.tar.gz" -mmin +1440 -exec rm -f {} \;

# Find and delete .html files under html/ older than 24 hours
find "$BASE_DIR"/html -type f -name "*.html" -mmin +1440 -exec rm -f {} \;
