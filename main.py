import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from pymining import seqmining
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the datasets
aisles = pd.read_csv('../dataset/aisles.csv')
departments = pd.read_csv('../dataset/departments.csv')
order_products_prior = pd.read_csv('../dataset/order_products__prior.csv',nrows=500)
order_products_train = pd.read_csv('../dataset/order_products__train.csv',nrows=500)
orders = pd.read_csv('../dataset/orders.csv',nrows=500)
products = pd.read_csv('../dataset/products.csv')

# Check for missing values in each dataset
print("Missing values in 'aisles':")
print(aisles.isnull().sum())

print("Missing values in 'departments':")
print(departments.isnull().sum())

print("Missing values in 'order_products__prior':")
print(order_products_prior.isnull().sum())

print("Missing values in 'order_products__train':")
print(order_products_train.isnull().sum())

print("Missing values in 'orders':")
print(orders.isnull().sum())

print("Missing values in 'products':")
print(products.isnull().sum())

# Check for duplicates in each dataset
print("Duplicate rows in 'aisles':")
print(aisles.duplicated().sum())

print("Duplicate rows in 'departments':")
print(departments.duplicated().sum())

print("Duplicate rows in 'order_products__prior':")
print(order_products_prior.duplicated().sum())

print("Duplicate rows in 'order_products__train':")
print(order_products_train.duplicated().sum())

print("Duplicate rows in 'orders':")
print(orders.duplicated().sum())

print("Duplicate rows in 'products':")
print(products.duplicated().sum())


# Assuming you have already loaded and preprocessed your data, and you have transactional data
# If not, you can follow the previous data preparation steps
aisles_df = pd.DataFrame(aisles)
departments_df = pd.DataFrame(departments)
order_products_prior_df = pd.DataFrame(order_products_prior)
order_products_train_df = pd.DataFrame(order_products_train)
orders_df = pd.DataFrame(orders)
products_df = pd.DataFrame(products)

# Example: Convert data into transactional data
# You can use your own logic to format the data appropriately
# In this example, 'transaction_data' should be a DataFrame with 'order_id' and 'product_name' columns
combined_data = pd.merge(order_products_prior, products, on='product_id', how='left')
combined_data = combined_data[['order_id', 'product_name']]

transaction_data = pd.DataFrame(combined_data)
transaction_data = transaction_data.groupby('order_id')['product_name'].apply(list).reset_index(name='items')

unique_products = list(set(product for sublist in transaction_data['items'] for product in sublist))

# Create an empty DataFrame with unique column names
oht_df = pd.DataFrame(columns=unique_products)

# Iterate through the data to set binary values
for idx, item_list in enumerate(transaction_data['items']):
    row = {product: 1 for product in item_list}
    oht_df = pd.concat([oht_df, pd.DataFrame(row, index=[idx])])

# Fill NaN values with 0
oht_df = oht_df.fillna(0)

# Apply the FP-Growth algorithm to mine frequent itemsets
frequent_itemsets = fpgrowth(oht_df, min_support=0.01, use_colnames=True)

# Determine a minimum support threshold (e.g., 0.01) to filter out infrequent itemsets
# Adjust the support threshold as needed based on your analysis goals

# Extract a list of frequent itemsets that meet the support threshold
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# Example: Print the frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Calculate confidence and lift values for each rule
# Confidence measures how often the rule has been found to be true.
# Lift measures the strength of association between antecedent and consequent.
# You can adjust the "min_threshold" parameter as needed.

# Filter the rules based on confidence and lift thresholds
min_confidence = 0.5  # Set your desired minimum confidence threshold
min_lift = 1.0  # Set your desired minimum lift threshold

filtered_rules = rules[(rules['confidence'] >= min_confidence) & (rules['lift'] >= min_lift)]

# Now you have a DataFrame 'filtered_rules' that contains the filtered association rules
# based on your confidence and lift thresholds.

# Example: Print the filtered association rules
print("Filtered Association Rules:")
print(filtered_rules)

# Convert transaction data into sequences (list of lists)
sequences = transaction_data['items'].tolist()

# Apply PrefixSpan algorithm for sequential pattern mining
max_pattern_length = 4  # Set the maximum desired pattern length
sequential_patterns = seqmining.freq_seq_enum(sequences, max_pattern_length)

# Save sequential patterns to a text file
with open('sequential_patterns.txt', 'w') as f:
    for pattern, _ in sequential_patterns:
        f.write(str(pattern) + '\n')

print("End of Sequential Pattern")
transaction_data['order_size'] = transaction_data['items'].apply(len)

# Select the features you want to use for clustering
features = transaction_data[['order_size']]

# Apply K-means clustering
num_clusters = 4  # Set the desired number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(features)

# Add cluster labels to the transaction data
transaction_data['cluster'] = cluster_labels

# Analyze the clusters
cluster_summary = transaction_data.groupby('cluster')['order_size'].describe()

# Print the cluster summary
print("Cluster Summary:")
print(cluster_summary)
