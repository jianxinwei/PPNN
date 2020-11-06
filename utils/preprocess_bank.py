import pandas as pd
import os


def toOneHot(df, col_name):
    new_df = df[col_name]
    new_df = pd.get_dummies(new_df, prefix=col_name)
    return new_df


bank_path = "../data/bank/bank.csv"
bank_full_path = "../data/bank/bank-full.csv"
bank = pd.read_csv(bank_full_path, sep=';',
                   usecols=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
                            "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
# print(bank)
# age, balance, duration, campaign, pdays, previous: [0,1]
# others: one-hot encoding
bank.drop(['contact', 'day', 'month'], axis=1, inplace=True)

numeric_col = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
# all_col = bank.columns.values.tolist()
all_col = bank.drop_duplicates().columns.values.tolist()

new_bank = pd.DataFrame(columns=[])
for c in all_col:
    if c in numeric_col:
        max_value = bank.loc[:, c].max()
        min_value = bank.loc[:, c].min()
        if max_value == min_value:
            continue
        new_col = bank[c].map(lambda x: (x - min_value) / (max_value - min_value))
        new_bank = pd.concat([new_bank, new_col], axis=1)
    elif c == 'y':
        new_col = bank[c].map(lambda x: 0 if x == 'no' else 1)
        new_bank = pd.concat([new_bank, new_col], axis=1)
    else:
        new_bank = pd.concat([new_bank, toOneHot(bank, c)], axis=1)

# print(new_bank.head())
# print(new_bank.columns.values.tolist())

new_bank.to_csv(os.path.join(os.path.split(bank_path)[0], 'new_bank_whole.csv'), sep=';', index=False)