import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' Loads the data from two separate sources and merges them together.

        Args:
        - messages_filepath (str): filepath/name of first csv file
        - categories_filepath (str): filepath/name of second csv file

        Returns: (df) merged dataframe
    '''
    df_msgs = pd.read_csv(messages_filepath)
    df_cats = pd.read_csv(categories_filepath)
    df_merged = pd.merge(df_msgs, df_cats, on='id', left_index=True, right_index=True)

    return df_merged


def clean_data(df):
    ''' Splits 'categories' column into separately named columns, conerts values to binary and 
        drops any duplicates (drops the original French column).

        Args:
        - df (df): merged df from load_data() function

        Returns (df): cleaned df
    '''
    # get category column headings
    cols = [i[:-2] for i in df['categories'][0].split(';')]

    # get binary values
    binaries = []
    for row in df['categories'].str.split(';'):
        binaries.append( [int(i[-1]) for i in row] )

    # put into temp df
    binary_df = pd.DataFrame(binaries, columns=cols)

    # merge with df and drop original 'categories' and French 'original' message columns 
    df_final = pd.concat([df.drop(['categories', 'original'], axis=1), binary_df], axis=1)

    # remove duplicate messages
    df_final = df_final.drop_duplicates(subset=['message'])

    #? [df_final[col].unique() for col in df_final]
    # remove rows where 'related' is non-binary (should only == 0 or 1; there are 188 rows == 2)
    df_final = df_final.drop(df_final[df_final['related'] == 2].index)
    ''' 'child_alone' columns has only zeros -- can we drop this or must we have 36 classes? '''

    return df_final


def save_data(df, database_filename):
    ''' Takes a cleaned dataframe and creates a sqlite database from it using SQLAlchemy.

        Args:
        - df (df): cleaned df from clean_data() function
        - database_filename (str): name given to newly created database

        Returns: None (db is saved to file: default='sqlite:///disaster_response.db')
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)

    return None  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()