import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class RecommendationSystem:
    def __init__(self, ratings):
        self.ratings = ratings

    def matrix_factorization(self, num_factors=10, learning_rate=0.01, reg=0.02, num_iterations=100):
        num_users, num_items = self.ratings.shape
        user_factors = np.random.normal(size=(num_users, num_factors))
        item_factors = np.random.normal(size=(num_items, num_factors))

        for _ in range(num_iterations):
            for user in range(num_users):
                for item in range(num_items):
                    if self.ratings[user, item] > 0:
                        error = self.ratings[user, item] - np.dot(user_factors[user], item_factors[item])
                        user_factors[user] += learning_rate * (error * item_factors[item] - reg * user_factors[user])
                        item_factors[item] += learning_rate * (error * user_factors[user] - reg * item_factors[item])

        return user_factors, item_factors

    def predict(self, user, item, user_factors, item_factors):
        return np.dot(user_factors[user], item_factors[item])

    def recommend(self, user, num_recommendations=10, user_factors=None, item_factors=None):
        if user_factors is None or item_factors is None:
            user_factors, item_factors = self.matrix_factorization()

        scores = [self.predict(user, item, user_factors, item_factors) for item in range(self.ratings.shape[1])]
        top_items = np.argsort(scores)[-num_recommendations:]
        return top_items

# Misol uchun ma'lumotlar
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

rs = RecommendationSystem(ratings)
recommended_items = rs.recommend(0)
print(recommended_items)
```

Bu kodda, `RecommendationSystem` klassi yaratilib, unda `matrix_factorization` metodi mavjud bo'lib, u oddiy algoritmga asoslangan matrix faktorizatsiyani amalga oshiradi. `predict` metodi mavjud bo'lib, u foydalanuvchi va mahsulot o'rtasidagi kabi hisoblashni amalga oshiradi. `recommend` metodi mavjud bo'lib, u foydalanuvchi uchun eng yuqori rekomendatsiyalarni qaytaradi. Misol uchun ma'lumotlar berilib, unda foydalanuvchi uchun eng yuqori rekomendatsiyalarni qaytaradi.
