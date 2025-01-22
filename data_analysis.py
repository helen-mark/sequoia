"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

from pandas import read_csv, DataFrame

dataset = read_csv('data/dataset.csv', delimiter=',')
dataset.head()
target_idx = 14  # index of "works/left" column

# Compute correlation matrix (correlation between each pair of variables):
corr_matrix = dataset.corr(method='kendall')fyfkbp lfyys[ b yfgbcfybt crhbgnjd]
print(corr_matrix)

np_corr_matrix = corr_matrix.to_numpy()
# Extract correlation between each variable vs target:
corr_with_target = np_corr_matrix[target_idx]

corr_dict = {}
for i, name in enumerate(corr_matrix):
    corr_dict[name] = corr_with_target[i]

for key in corr_dict:
    print(f'{corr_dict[key]}')
