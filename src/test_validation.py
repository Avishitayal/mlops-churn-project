from data_loader import load_data
from validate_data import validate_data


df = load_data()

validate_data(df)