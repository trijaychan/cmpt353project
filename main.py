import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from utils import *

# Run using the command:
#   python3 main.py data people.csv

def main():    
    directory = sys.argv[1]
    demographic_file = sys.argv[2]
    
    data = pd.read_csv(f"{directory}/{demographic_file}")
    data["steps_per_minute"] = data["name"].apply(lambda name: get_steps_per_minute(f"{directory}/{name.lower()}.csv"))
        
    # todo: testing
    
    males = data[data["gender"] == "M"]["steps_per_minute"]
    females = data[data["gender"] == "F"]["steps_per_minute"]
    
    _, p = mannwhitneyu(males, females)
    print(f"p-value: {p}")
    
if __name__ == "__main__":
    main()