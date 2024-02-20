import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the data containing timestamps and sentiment scores
data = pd.read_excel("D:/pythonProject1/数据尝试/twitter/output_with_sentiment_scores_twitter.xlsx")

# Convert the 'Timestamp' column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extract the month and year from the timestamp
data['Month'] = data['Timestamp'].dt.to_period('M').astype(str)  # Convert to string

# Convert the 'text_score' column to lists of floats and scale to [-1, 1]
data['text_score'] = data['text_score'].apply(lambda x: [(float(val) * 2) - 1 for val in x.strip('[]').split('][')])

# Calculate the average sentiment score for each month
monthly_avg_sentiment = data.groupby('Month')['text_score'].apply(lambda x: np.mean(np.concatenate(x.to_numpy()))).reset_index()

# Set the style using Seaborn
sns.set(style="whitegrid")

# Create a figure and axes
plt.figure(figsize=(20, 8))

# Plot sentiment over time using Seaborn's lineplot with a different color (e.g., blue)
sns.lineplot(x='Month', y='text_score', data=monthly_avg_sentiment, marker='o', color='slateblue')

# Add a polynomial fit curve
degree = 60  # Choose the degree of the polynomial (adjust as needed)
coefficients = np.polyfit(range(len(monthly_avg_sentiment)), monthly_avg_sentiment['text_score'], degree)
poly_fit = np.poly1d(coefficients)
x_range = np.arange(len(monthly_avg_sentiment))
plt.plot(x_range, poly_fit(x_range), color='green', label=f'Polynomial Fit (Degree {degree})')

# Add a bold line at y=0 position
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

# Add title and labels
plt.title('Monthly Average Sentiment Over Time with Polynomial Fit', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Sentiment Score', fontsize=14)

# Format the x-axis ticks for better readability
plt.xticks(rotation=45, ha='right')

# Display only every nth label on the x-axis (adjust n as needed)
n = 2
plt.xticks(range(0, len(monthly_avg_sentiment['Month']), n), monthly_avg_sentiment['Month'][::n])

# # Save the plot
# plt.savefig("monthly_average_sentiment_twitter_polyfit sentiment140.pdf")

# Show the plot
plt.legend()
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import numpy as np
from numpy.fft import fft, ifft

# Read the Twitter data containing timestamps and sentiment scores
twitter_data = pd.read_excel("D:/pythonProject1/数据尝试/twitter/output_with_sentiment_scores_twitter.xlsx")

# Convert the 'Timestamp' column to datetime
twitter_data['Timestamp'] = pd.to_datetime(twitter_data['Timestamp'])

# Extract the month and year from the timestamp
twitter_data['Month'] = twitter_data['Timestamp'].dt.to_period('M').astype(str)  # Convert to string

# Convert the 'text_score' column to lists of floats and scale to [-1, 1]
twitter_data['text_score'] = twitter_data['text_score'].apply(lambda x: [(float(val) * 2) - 1 for val in ast.literal_eval(x)])

# Calculate the average sentiment score for each month
monthly_avg_sentiment_twitter = twitter_data.groupby('Month')['text_score'].apply(
    lambda x: np.mean([item for sublist in x for item in sublist]) if len(x) > 0 else np.nan
).reset_index()
monthly_avg_sentiment_twitter.columns = ['Month', 'Score']

# Read the other data
other_data = pd.read_excel("D:/pythonProject1/数据尝试/狗屎/情感分析/output_with_sentiment_scores_CI only.xlsx")

# Convert the 'time' column to datetime
other_data['time'] = pd.to_datetime(other_data['time'])

# Convert the 'text_score' column from string representation of lists to actual lists
other_data['text_score'] = other_data['text_score'].apply(ast.literal_eval)

# Flatten the lists in the 'text_score' column and calculate the mean
other_data['average_text_score'] = other_data['text_score'].apply(lambda x: sum(x) / len(x) if x else None)

# Rescale the sentiment scores from [0, 1] to [-1, 1]
other_data['average_text_score'] = 2 * other_data['average_text_score'] - 1

# Extract the month and year from the timestamp
other_data['Month'] = other_data['time'].dt.to_period('M').astype(str)  # Convert to string

# Convert 'average_text_score' to numeric
other_data['average_text_score'] = pd.to_numeric(other_data['average_text_score'], errors='coerce')

# Sort the other data by 'Month'
other_data.sort_values(by='Month', inplace=True)

# Calculate the average sentiment score for each month
monthly_avg_sentiment_other = other_data.groupby('Month')['average_text_score'].mean().reset_index()

# Identify the overlapping time period
start_date = max(monthly_avg_sentiment_twitter['Month'].min(), monthly_avg_sentiment_other['Month'].min())
end_date = min(monthly_avg_sentiment_twitter['Month'].max(), monthly_avg_sentiment_other['Month'].max())

# Filter the data to the overlapping time period
twitter_data_overlap = monthly_avg_sentiment_twitter[
    (monthly_avg_sentiment_twitter['Month'] >= start_date) & (monthly_avg_sentiment_twitter['Month'] <= end_date)
]
other_data_overlap = monthly_avg_sentiment_other[
    (monthly_avg_sentiment_other['Month'] >= start_date) & (monthly_avg_sentiment_other['Month'] <= end_date)
]

# Set the style using Seaborn
sns.set(style="whitegrid")

# Create a figure and axes
plt.figure(figsize=(20, 8))

# Plot sentiment over time using Seaborn's lineplot with a different color (e.g., blue) for Twitter data
sns.lineplot(x='Month', y='Score', data=twitter_data_overlap, marker='o', color='lightblue', label='Twitter Data')

# Add a polynomial fit curve for Twitter data
degree_twitter = 60  # Choose the degree of the polynomial (adjust as needed)
coefficients_twitter = np.polyfit(
    range(len(twitter_data_overlap)), twitter_data_overlap['Score'], degree_twitter
)
poly_fit_twitter = np.poly1d(coefficients_twitter)
x_range_twitter = np.arange(len(twitter_data_overlap))
plt.plot(x_range_twitter, poly_fit_twitter(x_range_twitter), color='green', label=f'Twitter Data Poly Fit (Degree {degree_twitter})')

# Plot sentiment over time using Seaborn's lineplot with a different color (e.g., orange) for other data
sns.lineplot(x='Month', y='average_text_score', data=other_data_overlap, marker='o', color='tan', label='Other Data')

# Add a polynomial fit curve for other data
degree_other = 60  # Choose the degree of the polynomial (adjust as needed)
coefficients_other = np.polyfit(
    range(len(other_data_overlap)), other_data_overlap['average_text_score'], degree_other
)
poly_fit_other = np.poly1d(coefficients_other)
x_range_other = np.arange(len(other_data_overlap))
plt.plot(x_range_other, poly_fit_other(x_range_other), color='red', label=f'Other Data Poly Fit (Degree {degree_other})')

# Add title and labels
plt.title('Monthly Average Sentiment Overlapping Time Period', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Sentiment Score', fontsize=14)

# Format the x-axis ticks for better readability
plt.xticks(rotation=45, ha='right')

# Display only every nth label on the x-axis (adjust n as needed)
n = 2
plt.xticks(range(0, len(twitter_data_overlap['Month']), n), twitter_data_overlap['Month'][::n])

# # Save the plot
# plt.savefig("monthly_average_sentiment_overlap_polyfit_two.pdf")

# Show the plot
plt.legend()
plt.show()

# Cross-correlation analysis
def correlate(x, y, mode='full'):
    x = x - np.mean(x)
    y = y - np.mean(y)
    result = ifft(fft(x, len(x) * 2 - 1) * np.conj(fft(y, len(y) * 2 - 1)), len(x) * 2 - 1)
    if mode == 'full':
        return np.real(result)
    elif mode == 'valid':
        return np.real(result[len(x)-1:len(x)+len(y)-1])
    else:
        raise ValueError(f"Mode {mode} not supported")

# Identify the overlapping time period for cross-correlation
start_date_cross_corr = max(monthly_avg_sentiment_twitter['Month'].min(), other_data['Month'].min())
end_date_cross_corr = min(monthly_avg_sentiment_twitter['Month'].max(), other_data['Month'].max())

# Filter the data to the overlapping time period for cross-correlation
twitter_data_cross_corr = monthly_avg_sentiment_twitter[
    (monthly_avg_sentiment_twitter['Month'] >= start_date_cross_corr) & (monthly_avg_sentiment_twitter['Month'] <= end_date_cross_corr)
]
# Filter the data to the overlapping time period for cross-correlation
twitter_data_cross_corr = monthly_avg_sentiment_twitter[
    (monthly_avg_sentiment_twitter['Month'] >= start_date_cross_corr) & (monthly_avg_sentiment_twitter['Month'] <= end_date_cross_corr)
]
other_data_cross_corr = other_data[
    (other_data['Month'] >= start_date_cross_corr) & (other_data['Month'] <= end_date_cross_corr)
]

# Ensure both datasets have the same length for cross-correlation
min_length_cross_corr = min(len(twitter_data_cross_corr), len(other_data_cross_corr))
twitter_data_cross_corr = twitter_data_cross_corr.head(min_length_cross_corr)
other_data_cross_corr = other_data_cross_corr.head(min_length_cross_corr)

# Cross-correlation analysis
lags = np.arange(-min_length_cross_corr + 1, min_length_cross_corr)
corr_values = correlate(twitter_data_cross_corr['Score'], other_data_cross_corr['average_text_score'], mode='full')

# Find the index of the maximum correlation value
max_corr_index = np.argmax(corr_values)

# Extract the lag and corresponding correlation value at the maximum point
lag_at_max_corr = lags[max_corr_index]
max_corr_value = corr_values[max_corr_index]

# Plot the cross-correlation results with annotations
plt.figure(figsize=(12, 6))
plt.plot(lags, corr_values, marker='o', linestyle='-', color='blue')
plt.scatter(lag_at_max_corr, max_corr_value, color='red', marker='x', label='Max Correlation')

# Annotate the maximum correlation point
plt.annotate(f'Max Correlation: {max_corr_value:.2f} at Lag: {lag_at_max_corr} Months',
              xy=(lag_at_max_corr, max_corr_value), xytext=(lag_at_max_corr, max_corr_value + 0.5),
              arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

plt.title('Cross-Correlation of Twitter Data and Other Data')
plt.xlabel('Lag (Months)')
plt.ylabel('Cross-Correlation')
plt.legend()
plt.grid(True)
plt.show()


# Calculate the values of the polynomial fit curves for the overlapping time period
poly_fit_values_twitter = poly_fit_twitter(x_range_twitter)
poly_fit_values_other = poly_fit_other(x_range_other)

# Calculate the cross-correlation function between the two polynomial fit curves
cross_corr_poly_fit = np.correlate(poly_fit_values_twitter, poly_fit_values_other, mode='full')
lag_cross_corr_poly_fit = np.argmax(cross_corr_poly_fit) - len(poly_fit_values_twitter) + 1

print(f'Lag using cross-correlation of polynomial fit curves: {lag_cross_corr_poly_fit} points')

from scipy.signal import find_peaks

# Calculate the values of the polynomial fit curves for the overlapping time period
poly_fit_values_twitter = poly_fit_twitter(x_range_twitter)
poly_fit_values_other = poly_fit_other(x_range_other)

# Calculate the cross-correlation function between the two polynomial fit curves
cross_corr_poly_fit = np.correlate(poly_fit_values_twitter, poly_fit_values_other, mode='full')
lag_cross_corr_poly_fit = np.argmax(cross_corr_poly_fit) - len(poly_fit_values_twitter) + 1

print(f'Lag using cross-correlation of polynomial fit curves: {lag_cross_corr_poly_fit} points')

# Display only every nth label on the x-axis (adjust n as needed)
n = 2
plt.xticks(range(0, len(twitter_data_overlap['Month']), n), twitter_data_overlap['Month'][::n])

# Find peaks in the polynomial fit curves
peaks_twitter, _ = find_peaks(poly_fit_twitter(x_range_twitter))
peaks_other, _ = find_peaks(poly_fit_other(x_range_other))

# Calculate time difference (lag) between peaks
lag = x_range_twitter[peaks_twitter].mean() - x_range_other[peaks_other].mean()

print(f'Twitter data lags behind Other data by {lag} months')


# Plot the cross-correlation function
plt.figure(figsize=(10, 5))
lag_range = range(-len(poly_fit_values_twitter) + 1, len(poly_fit_values_twitter))
plt.plot(lag_range[:len(cross_corr_poly_fit)], cross_corr_poly_fit, marker='o')
plt.title('Cross-Correlation of Polynomial Fit Curves')
plt.xlabel('Lag (points)')
plt.ylabel('Cross-Correlation')
plt.grid(True)
plt.show()


# import numpy as np
# import pandas as pd
# import statsmodels.api as sm

# # Combine Twitter and Other data
# combined_data = pd.merge(twitter_data_overlap, other_data_overlap, on='Month', how='inner')

# # Define the SARIMAX model
# order = (10, 0, 10)  # You can adjust these order values based on model diagnostics
# seasonal_order = (10, 1, 10, 12)  # Add seasonal order for monthly data (12 months in a year)

# # Define exogenous variables
# exog_vars = combined_data['average_text_score']

# # SARIMAX model
# model_sarimax = sm.tsa.SARIMAX(combined_data['Score'], exog=exog_vars, order=order, seasonal_order=seasonal_order)

# # Fit the SARIMAX model
# results_sarimax = model_sarimax.fit()

# # Display model summary
# print(results_sarimax.summary())
# # 检查模型残差
# residuals = results_sarimax.resid

# # 绘制残差图
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# sns.histplot(residuals, kde=True)
# plt.title('Residuals Distribution')

# # 绘制残差的自相关图
# plt.subplot(1, 2, 2)
# sm.graphics.tsa.plot_acf(residuals, lags=40, alpha=0.05)
# plt.title('Residuals Autocorrelation')
# plt.show()

# # 进行 Ljung-Box 检验
# lb_test = sm.stats.acorr_ljungbox(residuals, lags=[20], return_df=True)
# print("Ljung-Box Test Results:")
# print(lb_test)

# # 观察滞后图
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# sm.graphics.tsa.plot_acf(residuals, lags=40, alpha=0.05)
# plt.title('Residuals Autocorrelation')

# # Ljung-Box检验
# plt.subplot(1, 2, 2)
# lb_test = sm.stats.acorr_ljungbox(residuals, lags=[20], return_df=True)
# print("Ljung-Box Test Results:")
# print(lb_test)

# # 绘制滞后图和Ljung-Box检验结果
# plt.show()

# # 滞后相关性分析
# lags = np.arange(-min_length_cross_corr + 1, min_length_cross_corr)
# corr_values = correlate(twitter_data_cross_corr['Score'], other_data_cross_corr['average_text_score'], mode='full')

# # 绘制滞后相关性图
# plt.figure(figsize=(12, 6))
# plt.plot(lags, corr_values, marker='o', linestyle='-', color='blue')
# plt.title('Cross-Correlation of Twitter Data and Other Data')

# # 找到最大相关性值的索引
# max_corr_index = np.argmax(corr_values)
# lag_at_max_corr = lags[max_corr_index]
# max_corr_value = corr_values[max_corr_index]

# # 在图中标记最大相关性点
# plt.scatter(lag_at_max_corr, max_corr_value, color='red', marker='x', label='Max Correlation')
# plt.annotate(f'Max Correlation: {max_corr_value:.2f} at Lag: {lag_at_max_corr} Months',
#               xy=(lag_at_max_corr, max_corr_value), xytext=(lag_at_max_corr, max_corr_value + 0.5),
#               arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

# plt.xlabel('Lag (Months)')
# plt.ylabel('Cross-Correlation')
# plt.legend()
# plt.grid(True)
# plt.show()