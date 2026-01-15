import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Изменяем общую тему на более современную 'ticks'
sns.set_theme(style="ticks")
OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    # Предполагаем наличие файла
    df = pd.read_csv('data/train.csv')
    return df

def plot_target_distribution(df):
    plt.figure(figsize=(11, 6))
    # 2. Используем цвет 'teal' и заполнение (fill=True) с другим стилем KDE
    sns.histplot(df['Calories'], kde=True, color='#2a9d8f', line_kws={'lw': 3}, alpha=0.6)
    
    plt.title('Distribution of Target Variable (Calories)', fontsize=15, pad=20)
    plt.xlabel('Calories Burned', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Убираем верхнюю и правую границы для "чистого" вида
    sns.despine()
    plt.savefig(f'{OUTPUT_DIR}/target_dist_calories.png', dpi=300)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/target_dist_calories.png")

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))
    
    numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
    if 'Sex' in df.columns:
        df_corr = df.copy()
        if df_corr['Sex'].dtype == 'O':
            df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
        numeric_cols.append('Sex')
        corr = df_corr[numeric_cols].corr()
    else:
        corr = df[numeric_cols].corr()

    # 3. Меняем цветовую карту на 'magma' или 'viridis' (они лучше воспринимаются)
    # Добавляем linewidths для разделения ячеек
    mask =  (corr == 1.0) # Можно добавить маску для треугольной матрицы, если нужно
    sns.heatmap(corr, annot=True, cmap='mako', fmt=".2f", center=0, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=300)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/correlation_matrix.png")

def plot_scatter_features(df):
    features = ['Duration', 'Heart_Rate', 'Body_Temp']
    # 4. Используем палитру 'rocket' и добавляем плотность распределения
    for feature in features:
        plt.figure(figsize=(9, 7))
        # Используем regplot для отображения линии тренда (помогает визуально)
        sns.regplot(x=df[feature], y=df['Calories'], 
                    scatter_kws={'alpha': 0.3, 'color': '#264653'}, 
                    line_kws={'color': '#e76f51'})
        
        plt.title(f'Relationship: {feature} vs Calories', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        sns.despine()
        
        plt.savefig(f'{OUTPUT_DIR}/scatter_{feature}_calories.png', dpi=300)
        plt.close()
        print(f"Saved {OUTPUT_DIR}/scatter_{feature}_calories.png")

if __name__ == "__main__":
    print("Starting enhanced visualization process...")
    try:
        df = load_data()
        plot_target_distribution(df)
        plot_correlation_matrix(df)
        plot_scatter_features(df)
        print("Done! All plots updated with new styles.")
    except Exception as e:
        print(f"Error: {e}")