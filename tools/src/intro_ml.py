# %% [markdown]
# # Introducción al aprendizaje automático
#
# En este cuaderno usted va a:
# 1. Comparar una regla escrita a mano con un modelo que aprende de ejemplos.
# 2. Ver el descenso por gradiente minimizar una pérdida.
# 3. Observar subajuste, buen ajuste y sobreajuste con polinomios de distinto grado.
#
# Ejecute las celdas en orden.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# %% [markdown]
# ## 1. Regla manual frente a modelo entrenado
#
# Generamos 400 correos ficticios con dos características numéricas
# (por ejemplo, proporción de mayúsculas y cantidad de enlaces).
# Primero escribimos una regla a mano y luego dejamos que un modelo encuentre la suya.

# %%
features, labels = make_classification(
    n_samples=400, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, class_sep=1.2, random_state=RANDOM_STATE,
)
train_x, test_x, train_y, test_y = train_test_split(
    features, labels, test_size=0.25, random_state=RANDOM_STATE
)


def hand_written_rule(samples: np.ndarray) -> np.ndarray:
    """Regla fija: si la primera característica supera 0, es spam."""
    return (samples[:, 0] > 0).astype(int)


def accuracy(predicted: np.ndarray, expected: np.ndarray) -> float:
    return float(np.mean(predicted == expected))


rule_accuracy = accuracy(hand_written_rule(test_x), test_y)
model = LogisticRegression().fit(train_x, train_y)
model_accuracy = accuracy(model.predict(test_x), test_y)
print(f"Regla manual: {rule_accuracy:.2%}")
print(f"Modelo entrenado: {model_accuracy:.2%}")

# %% [markdown]
# El modelo ajustó sus parámetros con los ejemplos de entrenamiento.
# La regla manual refleja solo la intuición de quien la escribió.

# %%
def plot_decision_regions(predict, samples, targets, title):
    grid_x, grid_y = np.meshgrid(
        np.linspace(samples[:, 0].min() - 1, samples[:, 0].max() + 1, 300),
        np.linspace(samples[:, 1].min() - 1, samples[:, 1].max() + 1, 300),
    )
    zone = predict(np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)
    plt.contourf(grid_x, grid_y, zone, alpha=0.2, cmap="coolwarm")
    plt.scatter(samples[:, 0], samples[:, 1], c=targets, cmap="coolwarm", edgecolor="k")
    plt.title(title)


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plot_decision_regions(hand_written_rule, test_x, test_y, "Regla manual")
plt.subplot(1, 2, 2)
plot_decision_regions(model.predict, test_x, test_y, "Modelo aprendido")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Descenso por gradiente desde cero
#
# Minimizamos la pérdida cuadrática de una recta $\hat{y}=w x + b$:
#
# $$J(w,b)=\frac{1}{n}\sum_i (y_i-wx_i-b)^2$$
#
# Los gradientes son
#
# $$\frac{\partial J}{\partial w}=-\frac{2}{n}\sum_i x_i\,(y_i-\hat y_i),\qquad
# \frac{\partial J}{\partial b}=-\frac{2}{n}\sum_i (y_i-\hat y_i)$$

# %%
TRUE_SLOPE, TRUE_INTERCEPT = 2.5, 1.0
x_values = rng.uniform(-3, 3, 60)
y_values = TRUE_SLOPE * x_values + TRUE_INTERCEPT + rng.normal(0, 1.0, 60)


def mean_squared_error(slope, intercept):
    return np.mean((y_values - slope * x_values - intercept) ** 2)


def gradient_step(slope, intercept, learning_rate):
    residual = y_values - (slope * x_values + intercept)
    slope_gradient = -2 * np.mean(x_values * residual)
    intercept_gradient = -2 * np.mean(residual)
    return slope - learning_rate * slope_gradient, intercept - learning_rate * intercept_gradient


def run_gradient_descent(learning_rate, steps=60):
    slope, intercept = -2.0, -3.0
    losses = [mean_squared_error(slope, intercept)]
    for _ in range(steps):
        slope, intercept = gradient_step(slope, intercept, learning_rate)
        losses.append(mean_squared_error(slope, intercept))
    return slope, intercept, losses


slope, intercept, losses = run_gradient_descent(learning_rate=0.05)
print(f"Pendiente aprendida: {slope:.2f} (real {TRUE_SLOPE})")
print(f"Intercepto aprendido: {intercept:.2f} (real {TRUE_INTERCEPT})")

# %%
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(x_values, y_values, alpha=0.7)
plt.plot(x_values, slope * x_values + intercept, "r", lw=2)
plt.title("Recta aprendida")
plt.subplot(1, 2, 2)
for rate in (0.005, 0.05, 0.12):
    plt.plot(run_gradient_descent(rate)[2], label=f"η = {rate}")
plt.yscale("log")
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.legend()
plt.title("Efecto de la tasa de aprendizaje")
plt.tight_layout()
plt.show()

# %% [markdown]
# Con $\eta=0.005$ la pérdida baja despacio. Con $\eta=0.12$ cae más rápido pero
# se acerca al límite de estabilidad. Un valor mayor haría que el método diverja.
#
# ## 3. Subajuste y sobreajuste
#
# Ajustamos polinomios de grado 1, 4 y 15 a una señal senoidal con ruido
# y comparamos el error en datos de entrenamiento y en datos nuevos.

# %%
def noisy_sine(sample_count):
    inputs = np.sort(rng.uniform(0, 1, sample_count))
    return inputs, np.sin(2 * np.pi * inputs) + rng.normal(0, 0.25, sample_count)


train_inputs, train_targets = noisy_sine(20)
test_inputs, test_targets = noisy_sine(200)


def polynomial_error(degree):
    coefficients = np.polyfit(train_inputs, train_targets, degree)
    train_error = np.mean((np.polyval(coefficients, train_inputs) - train_targets) ** 2)
    test_error = np.mean((np.polyval(coefficients, test_inputs) - test_targets) ** 2)
    return coefficients, train_error, test_error


dense_inputs = np.linspace(0, 1, 300)
plt.figure(figsize=(12, 3.8))
for position, degree in enumerate((1, 4, 15), start=1):
    coefficients, train_error, test_error = polynomial_error(degree)
    plt.subplot(1, 3, position)
    plt.scatter(train_inputs, train_targets, c="tab:blue")
    plt.plot(dense_inputs, np.polyval(coefficients, dense_inputs), "r")
    plt.ylim(-2, 2)
    plt.title(f"Grado {degree}\nentrenamiento {train_error:.2f} | prueba {test_error:.2f}")
plt.tight_layout()
plt.show()

# %% [markdown]
# El grado 15 casi no tiene error de entrenamiento, pero su error en datos nuevos es
# mucho mayor. Esa diferencia es la señal típica de sobreajuste.

# %%
degrees = range(1, 16)
errors = [polynomial_error(d)[1:] for d in degrees]
plt.plot(degrees, [e[0] for e in errors], "o-", label="entrenamiento")
plt.plot(degrees, [e[1] for e in errors], "o-", label="prueba")
plt.yscale("log")
plt.xlabel("Grado del polinomio")
plt.ylabel("Error cuadrático medio")
plt.legend()
plt.show()

# %% [markdown]
# ## Ejercicios
#
# 1. Cambie `class_sep` a `0.5` en `make_classification`. ¿Qué pasa con la precisión de ambos métodos?
# 2. Pruebe `learning_rate=0.2` en `run_gradient_descent`. ¿Qué observa en la pérdida?
# 3. Use 100 puntos de entrenamiento en lugar de 20. ¿Sigue siendo tan malo el grado 15?
