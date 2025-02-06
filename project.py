import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df= pd.read_csv('C:/Users/subha/OneDrive/Desktop/e commerce sales analysis.csv')
print(df)

#data cleaning
# Step 1: Check for missing values
missing_values = df.isnull().sum()

# Step 2: Remove duplicates
data_cleaned = df.drop_duplicates()

# Step 3: Ensure correct data types
data_cleaned['price'] = pd.to_numeric(data_cleaned['price'], errors='coerce')
data_cleaned['review_score'] = pd.to_numeric(data_cleaned['review_score'], errors='coerce')

# Step 4: Check for outliers (basic check by describing numerical columns)
data_description = data_cleaned.describe()

# Step 5: Check cleaned data
data_cleaned_info = data_cleaned.info()
print(missing_values)
missing_values, data_cleaned_info, data_description
df['Total Sales'] = df[['sales_month_1', 'sales_month_2', 'sales_month_3',
       'sales_month_4', 'sales_month_5', 'sales_month_6', 'sales_month_7',
       'sales_month_8', 'sales_month_9', 'sales_month_10', 'sales_month_11',
       'sales_month_12']].sum(axis = 1)
df.head()

#1 Average Price for each category plot-1
category_avg_price = df.groupby('category')['price'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(8,5))
ax = sns.scatterplot(x='price', y='category', data=category_avg_price, palette='Blues_d')
# data labels
for index, value in enumerate(category_avg_price['price']):
   plt.text(value, index, f'{value:.2f}', va='center', ha='left', color='black',fontsize = 10)

plt.title('Average Price of Each Category',weight = 'bold',fontsize =16)
plt.xlabel('Average Price',fontsize =14)
plt.ylabel('Category',fontsize =14)
ax.set_xlim(220,280)
plt.show()
#2 Analyze sales trends over the 12 months plot-2
sales_columns = [f'sales_month_{i}' for i in range(1,13)]
monthly_sales = df[sales_columns].sum()

# Plot the line chart
plt.figure( figsize=(8,5))
monthly_sales.plot(kind='line', color= 'darkblue')
plt.xlabel('Month',fontsize = 14)
plt.ylabel('Total Sales',fontsize = 14)
plt.title( 'Monthly Sales Trends',weight = 'bold',fontsize = 16)
plt.xticks(rotation =45)
plt.tight_layout()
plt.grid(True)
plt.show()

#3 Review score distribution plot-3

plt.figure(figsize = (6,4))
# Histogram
sns.histplot(data = df, x= 'review_score', bins = 'auto')
plt.title("Review Score Distribution", weight = 'bold')

plt.tight_layout()
plt.show()

#4 bar Plot for the Total Sales per product plot-4
plt.figure(figsize=(12,6))
plt.bar('product_name','Total Sales',data = df, color='skyblue')
plt.xlabel('Product Name',fontsize = 16)
plt.ylabel('Total Sales',fontsize = 16)
plt.title('Total Sales per Product',weight = 'bold',fontsize = 18)
plt.xticks(rotation=90)
plt.show()

#5 sales performance for each category plot-5

df['total_sales'] = df.loc[:, 'sales_month_1':'sales_month_12'].sum(axis=1)
category_sales = df.groupby('category')['total_sales'].sum().reset_index()
category_sales = category_sales.sort_values(by='total_sales', ascending=False)
#visual
plt.figure(figsize=(10, 6))
sns.barplot(data=category_sales, x='category', y='total_sales', palette='viridis')
plt.title('Sales performance from each category')
plt.grid(True)

#6 scatterplot total sales vs price plot-6
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='total_sales', y='price', hue='category')
plt.title('Total Sales vs Price')

#7 piecahrt Distribution of categories plot-7
cat_counts = df['category'].value_counts().reset_index().sort_values(by='category')

plt.figure(figsize=(8, 8))
colors = sns.color_palette('Set2', len(cat_counts))
plt.pie(cat_counts['count'], labels=cat_counts['category'], autopct='%.0f%%', startangle=140, colors=colors)
plt.title('Distribution of Categories',fontweight='bold', fontsize=14)

#8 correlation matrix plot-8
# Impact of the price on sales and review scores

corr_matrix = df[['price', 'total_sales', 'review_score']].corr()
print("Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
# Step 1: Prepare the data for prediction
# Using 'price' and 'review_score' to predict 'total_sales'
# Select features and target variable
X = df[['price', 'review_score']]  # Features
y = df['total_sales']  # Target (total sales)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Building - Linear Regression
model = LinearRegression()

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Make predictions using the test data
y_pred = model.predict(X_test)

# Step 6: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the actual vs predicted sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

