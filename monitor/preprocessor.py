import json
import random

# sample preprocess_handler (to be implemented by customer)
# This is a trivial example, where we simply generate random values
# But customers can read the data from inference_record and trasnform it into 
# a flattened json structure

# NOTE: The purpose of this function is to ensure that the model input and output
# data has been flattened into a dictionary (with keys matching the column names
# from the baselining job. Model Monitor cannot process the data if it is not flat.
#
def preprocess_handler(inference_record):
    event_data = inference_record.event_data
    input_data = {}
    output_data = {}

    input_data['feature0'] = random.randint(1, 3)
    input_data['feature1'] = random.uniform(0, 1.6)
    input_data['feature2'] = random.uniform(0, 1.6)

    output_data['prediction0'] = random.uniform(1, 30)
    
    return {**input_data, **output_data}

