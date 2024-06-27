import pandas as pd

def xlsx_to_csv(xlsx_file_path, csv_file_path):
    """
    Converts an Excel file (.xlsx) to a CSV file (.csv), renaming columns
    to match the swim_competition_results table schema.

    Args:
        xlsx_file_path (str): The path to the Excel file.
        csv_file_path (str): The path to save the CSV file.
    """

    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(xlsx_file_path)

        # Rename columns to match the database table
        column_mapping = {
            # Map existing column names to the desired names
            '賽會名稱': 'competition_name',
            '項次': 'item_number',
            '單位名稱': 'unit_name',
            '姓名': 'name',
            '性別': 'gender',
            '比賽項目': 'event',
            '比賽成績': 'performance',
            '狀態': 'status'
        }
        df = df.rename(columns=column_mapping)

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False) 

        print(f"Conversion successful! CSV file saved at: {csv_file_path}")

    except FileNotFoundError:
        print(f"Error: Excel file not found at: {xlsx_file_path}")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    xlsx_file_path = '成績資料匯出.xlsx'
    csv_file_path = 'grades.csv'
    xlsx_to_csv(xlsx_file_path, csv_file_path)
