# Personalized Weighted Slope One Recommender System

## Overview
This project implements and evaluates a Personalized Weighted Slope One recommender system, which enhances the original Weighted Slope One algorithm by incorporating user similarity metrics. This approach combines the simplicity and efficiency of the Slope One family of algorithms with the personalization capabilities of neighborhood-based collaborative filtering.

## Background
Recommendation systems are fundamental tools in modern applications, from e-commerce to content streaming platforms. The Slope One algorithm, introduced by Lemire and Maclachlan, offers a computationally efficient approach to collaborative filtering with reasonable accuracy. This project extends their work by adding personalization through user similarity metrics.

## Key Features
- Implementation of the original Slope One and Weighted Slope One algorithms
- Enhanced personalization through user similarity metrics
- Centered Cosine Similarity implementation for user similarity calculations
- Parameter tuning capabilities for optimizing recommendation quality
- Comprehensive evaluation metrics

## Algorithm Details

The Personalized Weighted Slope One algorithm modifies the original Weighted Slope One method by incorporating user similarity in the deviation calculation:

```
devj,i = λ * (sum of (uj - ui) / card(Sj,i(χ))) + 
         (1-λ) * (sum of ((uj - ui) * exp(sim(u,u'))) / sum of (exp(sim(u,u')) * card(Sj,i(χ))))
```

Where:
- λ is a parameter between 0 and 1 that controls the influence of user similarity
- Sj,i(χ) is the set of all evaluations containing both items i and j
- sim(u,u') is the Centered Cosine Similarity between users
- exp() is the exponential function with base 2

The final prediction is calculated as:

```
PpwSl(u')j = sum of ((devj,i + u'i) * cj,i) / sum of cj,i
```

Where cj,i represents the number of users who rated both items j and i.

## Implementation

The implementation includes:
- Data preprocessing and manipulation functions
- User similarity calculation with Centered Cosine Similarity
- Deviation calculation incorporating user similarity
- Final prediction computation
- Parameter tuning for optimal λ value
- Evaluation metrics calculation

## Results

The implementation demonstrates that:
1. Personalization through user similarity improves recommendation quality
2. The λ parameter allows for fine-tuning the algorithm's performance
3. The approach maintains the computational efficiency of the original Slope One family

## Usage

```python
# Example usage
import pandas as pd
from personalized_slope_one import PersonalizedWeightedSlopeOne

# Load your rating data
# Format: user_id, item_id, rating
ratings_df = pd.read_csv('your_ratings_data.csv')

# Initialize the recommender
recommender = PersonalizedWeightedSlopeOne(lambda_param=0.5)

# Train the model
recommender.fit(ratings_df)

# Make predictions for a specific user
user_id = 42
predictions = recommender.predict(user_id)
```

## Requirements
- Python 3.6+
- NumPy
- Pandas
- Matplotlib (for visualization)
- Jupyter (for notebooks)

## Installation

```bash
git clone https://github.com/yourusername/personalized-weighted-slope-one-recommender.git
cd personalized-weighted-slope-one-recommender
pip install -r requirements.txt
```

## Future Work
- Implementing more efficient data structures for large-scale applications
- Exploring different similarity metrics
- Incorporating implicit feedback
- Extending to context-aware recommendations

## License
MIT License

## Acknowledgments
This implementation is based on concepts from the paper "Slope One Predictors for Online Rating-Based Collaborative Filtering" by Daniel Lemire and Anna Maclachlan.
