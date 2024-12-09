import pandas as pd


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def print_df_describe(df: pd.DataFrame, 
                      name: str,
                      show: bool = True):
    # Get an overview of the data
    if show:
        print(f'{color.BLUE}{name}{color.END} (head of the data is):')
        print(df.head())
        print()

        # Get an describe of the data
        print(f'{color.BLUE}{name}{color.END} (describe of the data is):')
        print(df.describe())
        print()

        #　Checking the shape of the dataframes
        print(f"Shape of {name} : {color.BLUE}{df.shape}{color.END}")
        print('-' * 80)

        # Get an isnull of the data
        print(f'{color.BLUE}{name}{color.END} (is null of the data is):')
        print(df.isnull().sum())
        print()

        # Get an isna of the data
        print(f'{color.BLUE}{name}{color.END} (is nan of the data is):')
        print(df.isna().sum())
        print()

        # Get an isna of the data
        print(f'{color.BLUE}{name}{color.END} (duplicated of the data is):')
        print(df.duplicated().sum())

        # Get info from data
        print(f'{color.BLUE}{name}{color.END} (info of the data is):')
        print(df.info())
        

if __name__ == '__main__':
   pass