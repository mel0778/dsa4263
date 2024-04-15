import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()
# username = os.getenv('sql_username')
# password = os.getenv('sql_password')
# localhost = os.getenv('sql_localhost')
# sqlport = os.getenv('sql_port')
# dbname = os.getenv('sql_dbname')

username = 'guest'
password = 'guest'
localhost = 'postgresdb'
sqlport = 5432
dbname = 'defaultdb'

engine = create_engine(
    f'postgresql://{username}:{password}@{localhost}:{sqlport}/{dbname}')
df = pd.read_csv('test.csv')
df.to_sql('CERT', engine)
