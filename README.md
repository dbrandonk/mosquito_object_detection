## Project Brief
https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023

## Installation
To install all the necessary dependencies, please run the following command at the root of the project:

```
pip install .
```

## Getting Started
To begin working with this project, you need to obtain the training data from the challenge and add it to the data directory of this project. Follow the steps below:

1. Set your API key as an environment variable:
   ```
   API_KEY="XXX"  # Replace with your actual API key
   ```

2. Log in to the AICrowd platform using the CLI:
   ```
   aicrowd login --api-key $API_KEY
   ```

3. Create a directory named 'data' and navigate into it:
   ```
   mkdir data && cd data
   ```

4. Download the challenge dataset using the AICrowd CLI:
   ```
   aicrowd dataset download --challenge mosquitoalert-challenge-2023
   ```

## Entry Points
TBD

## TODO
- Finish.
