import pandas as pd
import os


def toOneHot(df, col_name):
    new_df = df[col_name]
    new_df = pd.get_dummies(new_df, prefix=col_name)
    return new_df


bidding_path = "../data/bidding/Shill Bidding Dataset.csv"

bidding_data = pd.read_csv(bidding_path, sep=',')
bidding_data.drop(['Record_ID', 'Auction_ID', 'Bidder_ID'], axis=1, inplace=True)
print(bidding_data.head())

onehot_col = ['Successive_Outbidding']
all_col = bidding_data.columns.values.tolist()

new_bidding = pd.DataFrame(columns=[])
for c in all_col:
    if c in onehot_col:
        new_bidding = pd.concat([new_bidding, toOneHot(bidding_data, c)], axis=1)
    elif c == 'Class':
        new_bidding = pd.concat([new_bidding, bidding_data[c]], axis=1)
        new_bidding.rename(columns={'Class': 'y'}, inplace=True)
    else:
        max_value = bidding_data.loc[:, c].max()
        min_value = bidding_data.loc[:, c].min()
        if max_value == min_value:
            continue
        new_col = bidding_data[c].map(lambda x: (x - min_value) / (max_value - min_value))
        new_bidding = pd.concat([new_bidding, new_col], axis=1)

print(new_bidding.head())
# print(new_bidding.columns.values.tolist())
new_bidding.to_csv(os.path.join(os.path.split(bidding_path)[0], 'new_bidding.csv'), sep=';', index=False)