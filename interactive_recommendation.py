import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# === Step 1: Movie Data ===
movies_data = {
    'title': [
        'The Matrix', 'John Wick', 'Inception', 
        'The Dark Knight', 'Interstellar', 'Avengers'
    ],
    'genre': [
        'Action Sci-Fi', 'Action Thriller', 'Sci-Fi Thriller',
        'Action Crime', 'Sci-Fi Drama', 'Action Sci-Fi'
    ]
}
movies_df = pd.DataFrame(movies_data)

# === Step 2: User Ratings ===
ratings_data = {
    'The Matrix': [5, 4, 0, 5],
    'John Wick': [4, 0, 0, 4],
    'Inception': [0, 5, 4, 0],
    'The Dark Knight': [5, 4, 4, 0],
    'Interstellar': [0, 4, 5, 0],
    'Avengers': [4, 0, 0, 5]
}
users = ['User1', 'User2', 'User3', 'User4']
ratings_df = pd.DataFrame(ratings_data, index=users)

# === Step 3: Content-Based Similarity ===
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies_df['genre'])
content_similarity = cosine_similarity(genre_matrix)

# === Step 4: Collaborative Filtering ===
user_similarity = cosine_similarity(ratings_df.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=users, columns=users)

# === Step 5: Hybrid Recommendation Function ===
def hybrid_recommend(user_name, movie_name, top_n=3):
    if movie_name not in movies_df['title'].values:
        return ["Movie not found!"]
    if user_name not in ratings_df.index:
        return ["User not found!"]
    
    # Content similarity
    movie_idx = movies_df[movies_df['title'] == movie_name].index[0]
    content_scores = list(enumerate(content_similarity[movie_idx]))
    
    # Collaborative similarity
    similar_users = user_similarity_df[user_name].sort_values(ascending=False)[1:]
    most_similar_user = similar_users.index[0]
    
    liked_movies = ratings_df.loc[most_similar_user][ratings_df.loc[most_similar_user] >= 4].index.tolist()
    
    # Combine
    combined_scores = []
    for idx, score in content_scores:
        title = movies_df.iloc[idx]['title']
        bonus = 0.5 if title in liked_movies else 0
        combined_scores.append((title, score + bonus))
    
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    return [title for title, _ in combined_scores[1:top_n+1]]

# === Step 6: Interactive Terminal App ===
print("🎬 Welcome to the Movie Recommendation System! 🎬")
print("Available movies:")
for m in movies_df['title']:
    print("-", m)

# Get user input
user_name = input("\nEnter your username (User1, User2, User3, User4): ").strip()
movie_choice = input("Enter your favorite movie from the list above: ").strip()

# Show recommendations
recommendations = hybrid_recommend(user_name, movie_choice)
print(f"\n🎯 Recommendations for {user_name} based on '{movie_choice}':")
for rec in recommendations:
    print("-", rec)
