from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# بارگذاری مجموعه داده
iris = load_iris()

X = iris.data
y = iris.target

# تقسیم داده‌ها به آموزش و آزمایش
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ساخت مدل
model = KNeighborsClassifier(n_neighbors=3)

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی
predictions = model.predict(X_test)

# محاسبه دقت
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# پیش‌بینی یک گل جدید
new_flower = [[5.1, 3.5, 1.4, 0.2]]

result = model.predict(new_flower)

print("Predicted Flower:", iris.target_names[result[0]])