import pandas as pd
import os


def toOneHot(df, col_name):
    new_df = df[col_name]
    new_df = pd.get_dummies(new_df, prefix=col_name)
    return new_df


credit_path = "../data/credit/default of credit card clients.xls"

credit_data = pd.read_excel(credit_path, header=1)
credit_data.drop(['ID'], axis=1, inplace=True)
# print(credit_data.head())

numeric_col = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
               'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
all_col = credit_data.columns.values.tolist()

new_credit = pd.DataFrame(columns=[])
for c in all_col:
    if c in numeric_col:
        max_value = credit_data.loc[:, c].max()
        min_value = credit_data.loc[:, c].min()
        if max_value == min_value:
            continue
        new_col = credit_data[c].map(lambda x: (x - min_value) / (max_value - min_value))
        new_credit = pd.concat([new_credit, new_col], axis=1)
    elif c == 'default payment next month':
        new_credit = pd.concat([new_credit, credit_data[c]], axis=1)
        new_credit.rename(columns={'default payment next month': 'y'}, inplace=True)
    else:
        new_credit = pd.concat([new_credit, toOneHot(credit_data, c)], axis=1)

print(new_credit.head())
# print(new_credit.columns.values.tolist())
new_credit.to_csv(os.path.join(os.path.split(credit_path)[0], 'new_credit.csv'), sep=';', index=False)
