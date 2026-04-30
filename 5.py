import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('emails.csv')
    
    # 2. Basic Dataset Overview
    print("\n--- Basic Dataset Overview ---")
    print(f"Total number of emails (rows): {df.shape[0]}")
    print(f"Total number of tracked words/columns: {df.shape[1]}")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Total missing values in dataset: {missing_values}")

    # 3. Data Cleaning
    # Separate the identifier column and the target column (Prediction) if it exists
    df_features = df.copy()
    
    if 'Email No.' in df_features.columns:
        df_features = df_features.drop(['Email No.'], axis=1)
        
    if 'Prediction' in df_features.columns: # Often the last column in this dataset
        df_features = df_features.drop(['Prediction'], axis=1)

    # 4. Word Frequency Analysis (Top 20 most used words)
    print("\n--- Word Frequency Analysis ---")
    # Sum up the frequencies of each word across all emails
    word_counts = df_features.sum().sort_values(ascending=False)
    
    print("Top 10 most frequent words overall:")
    print(word_counts.head(10))

    # Plot the Top 20 most frequent words
    plt.figure(figsize=(12, 6))
    sns.barplot(x=word_counts.head(20).values, y=word_counts.head(20).index, palette="viridis")
    plt.title("Top 20 Most Frequent Words Across All Emails", fontsize=16)
    plt.xlabel("Total Count", fontsize=12)
    plt.ylabel("Words", fontsize=12)
    plt.tight_layout()
    plt.show()

    # 5. Email Length Analysis (Total word counts per email)
    print("\n--- Email Length Analysis ---")
    # Sum across the columns for each row to get the total words recorded per email
    df['Total_Words'] = df_features.sum(axis=1)
    
    print(f"Average words per email: {df['Total_Words'].mean():.2f}")
    print(f"Max words in a single email: {df['Total_Words'].max()}")
    print(f"Min words in a single email: {df['Total_Words'].min()}")

    # Plot the distribution of email lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Total_Words'], bins=50, kde=True, color="coral")
    plt.title("Distribution of Total Tracked Words per Email", fontsize=16)
    plt.xlabel("Total Tracked Words", fontsize=12)
    plt.ylabel("Number of Emails", fontsize=12)
    
    # Zoom in on the majority of the data to avoid heavily skewed plots from massive outliers
    plt.xlim(0, df['Total_Words'].quantile(0.95)) 
    plt.tight_layout()
    plt.show()

    # 6. Analyze Spam vs. Ham (If 'Prediction' column exists)
    if 'Prediction' in df.columns:
        print("\n--- Spam vs. Ham Distribution ---")
        spam_counts = df['Prediction'].value_counts()
        print("0 = Not Spam (Ham), 1 = Spam")
        print(spam_counts)
        
        plt.figure(figsize=(6, 6))
        plt.pie(spam_counts, labels=['Not Spam', 'Spam'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90)
        plt.title("Spam vs Not Spam Distribution")
        plt.show()

if __name__ == "__main__":
    main()
