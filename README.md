# Docker Branch Readme

## Overview

This branch contains Docker configurations tailored for specific tasks related to our project. It's designed with a focus on exporting model weights efficiently. This README provides guidance on how to utilize this Docker configuration effectively.

## Purpose

The primary objective of this Docker setup is to streamline the process of exporting model weights. It's optimized for these tasks rather than exploratory data analysis (EDA), which we believe is better suited for notebook environments (refer to prod branch for this).

## Contents

- **Processed Data Files**: We've retained processed data files within this branch. These files are relatively smaller in size, enabling faster loading and extraction compared to the entire dataset.
- **Python Scripts for Models**: Can be found in python_scripts but to understand what is happening step by step it is best to refer to prod branch.

## Usage

To utilize this Docker configuration effectively, follow these steps:

1. **Clone the Repository**: Start by cloning this repository to your local machine.

`git clone <repo link>`

2 . Switch to Docker Branch: Move to the Docker branch of the repository.
`git checkout docker`

3.  **Build Docker Image**: Build the Docker image using the provided Docker Compose file.
    `docker compose up`

## Notes

Kept the processed data files since they are of smaller size and allows for the skipping of loading in the entire dataset and extracting it which is a significant time and space task, just run this task on the prod branch instead using the provided script in python_scripts\data which is a more interactive experience. Once again, for exploratory data analysis (EDA) and similar tasks, we recommend referring to the `prod` branch or utilizing notebook environments for a more interactive experience since, the purpose of this docker file is to generate the figures and export model weights rather than for EDA purposes
