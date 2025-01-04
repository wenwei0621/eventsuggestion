import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the activity data
try:
    activity_data = pd.read_csv('Events.csv')  # Replace with your file name
    activity_data.columns = activity_data.columns.str.strip()  # Clean column names
    activity_data['Participants'] = activity_data['Participants'].astype(int)  # Ensure numeric type for Participants
except Exception as e:
    raise RuntimeError(f"Failed to load activity data: {e}")

# One-hot encode features for cosine similarity
activity_features = pd.get_dummies(activity_data[['Duration', 'Category', 'Participants', 'SDG']])

# In-memory storage to track user sessions and suggested activities
session_store = {}

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "Input data is required"}), 400

        # Ensure at least 2 features are provided
        required_features = ['Duration', 'Category', 'Participants', 'SDG']
        selected_features = [key for key in required_features if key in input_data and input_data[key]]

        if len(selected_features) < 2:
            return jsonify({"error": "Please select at least 2 features for filtering."}), 400

        # Create a unique session ID based on input data (you can use client-specific identifiers)
        session_id = str(input_data)

        # Filter the dataset based on selected features
        filtered_data = activity_data.copy()
        strict_match_count = 0

        for feature in selected_features:
            filtered_data = filtered_data[filtered_data[feature] == input_data[feature]]
            strict_match_count += 1

            # Stop filtering after two strict matches
            if strict_match_count == 2:
                break

        # If no matching activities are found, use the entire dataset as a fallback
        if filtered_data.empty:
            filtered_data = activity_data

        # One-hot encode input data for similarity scoring
        input_df = pd.DataFrame([input_data])
        input_features = pd.get_dummies(input_df)
        input_features = input_features.reindex(columns=activity_features.columns, fill_value=0)

        # Calculate cosine similarity with the filtered (or entire) dataset
        filtered_features = pd.get_dummies(filtered_data[['Duration', 'Category', 'Participants', 'SDG']])
        filtered_features = filtered_features.reindex(columns=activity_features.columns, fill_value=0)

        similarity_scores = cosine_similarity(input_features, filtered_features)

        # Rank activities by weighted similarity
        weights = {'Category': 4, 'Duration': 2, 'Participants': 2, 'SDG': 1}
        weighted_scores = (
            similarity_scores[0] *
            filtered_features.mul(
                filtered_features.columns.map(
                    lambda col: weights.get(col.split('_')[0], 1)
                ),
                axis=1
            ).sum(axis=1).values
        )

        # Track already suggested indices for this session
        if session_id not in session_store:
            session_store[session_id] = set()  # Initialize session storage

        # Exclude previously suggested activities
        available_indices = [i for i in range(len(filtered_data)) if i not in session_store[session_id]]

        # If no more activities are available, reset the session for cyclic suggestions
        if not available_indices:
            session_store[session_id].clear()  # Reset the session store
            available_indices = list(range(len(filtered_data)))

        # Get top 5 activities based on similarity
        top_indices = sorted(available_indices, key=lambda i: weighted_scores[i], reverse=True)[:5]
        matched_activities = filtered_data.iloc[top_indices].to_dict(orient='records')

        # If fewer than 5 activities are matched, fill the remainder with fallback activities
        if len(matched_activities) < 5:
            fallback_data = activity_data[~activity_data.index.isin(filtered_data.index)]
            fallback_features = pd.get_dummies(fallback_data[['Duration', 'Category', 'Participants', 'SDG']])
            fallback_features = fallback_features.reindex(columns=activity_features.columns, fill_value=0)
            fallback_similarity_scores = cosine_similarity(input_features, fallback_features)
            fallback_top_indices = fallback_similarity_scores[0].argsort()[::-1][: (5 - len(matched_activities))]
            fallback_activities = fallback_data.iloc[fallback_top_indices].to_dict(orient='records')
            matched_activities.extend(fallback_activities)

        # Update session store with the indices of suggested activities
        session_store[session_id].update(top_indices)

        return jsonify(matched_activities)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
