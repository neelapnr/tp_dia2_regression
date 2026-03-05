import numpy
import pandas
import matplotlib.pyplot as plt
import random

random.seed(1)




def plot(problems, solutions, predictions):
	plt.figure(figsize=(15, 6))
	plt.plot(problems, solutions, "ro")
	plt.plot(problems, predictions, "g")
	plt.show()


def plot_rmse(rmse_values_per_epoch):
	plt.figure(figsize=(15, 6))
	plt.plot(rmse_values_per_epoch, "b")
	plt.xlabel("epochs")
	plt.ylabel("rmse")
	plt.show()


def plot_comparison(problems, solutions, predictions_linear, predictions_quadratic):
	plt.figure(figsize=(15, 6))
	plt.plot(problems, solutions, "ro", label="donnees")
	plt.plot(problems, predictions_linear, "b", label="lineaire")
	plt.plot(problems, predictions_quadratic, "g", label="quadratique")
	plt.legend()
	plt.show()




def quadratic_regression(a, b, c, problems):
	return a * problems**2 + b * problems + c


def linear_regression(a, b, problems):
	return a * problems + b




def compute_rmse(predictions, solutions):
	n = len(predictions)
	return (1/n) * numpy.sqrt(numpy.sum(((predictions - solutions))**2))




def backpropagation_quadratic(a, b, c, n, problems, solutions, learning_rate, visual=False):
	
	errors = a * problems**2 + b * problems + c - solutions

	dmse_da = 2/n * sum(errors * problems**2)
	dmse_db = 2/n * sum(errors * problems)
	dmse_dc = 2/n * sum(errors)

	
	a -= learning_rate * dmse_da
	b -= learning_rate * dmse_db
	c -= learning_rate * dmse_dc

	predictions = quadratic_regression(a, b, c, problems)
	rmse = compute_rmse(predictions, solutions)

	if visual:
		plot(problems=problems, solutions=solutions, predictions=predictions)

	return a, b, c, rmse




def backpropagation(a, b, n, problems, solutions, learning_rate, visual=False):
	dmse_da = 2/n * sum((a * problems + b - solutions) * problems)
	dmse_db = 2/n * sum(a * problems + b - solutions)

	a -= learning_rate * dmse_da
	b -= learning_rate * dmse_db

	predictions = linear_regression(a, b, problems)
	rmse = compute_rmse(predictions, solutions)

	if visual:
		plot(problems=problems, solutions=solutions, predictions=predictions)

	return a, b, rmse




def gradient_descent_quadratic(problems, solutions, learning_rate=10**(-3), epochs=100):
	a, b, c = random.random(), random.random(), random.random()

	n = len(problems)

	rmse_values_per_epoch = []

	for index_epoch in range(epochs):
		a, b, c, rmse = backpropagation_quadratic(a, b, c, n, problems, solutions, learning_rate)
		rmse_values_per_epoch.append(rmse)

	predictions = quadratic_regression(a, b, c, problems)
	plot(problems=problems, solutions=solutions, predictions=predictions)

	return a, b, c, rmse_values_per_epoch



def gradient_descent(problems, solutions, learning_rate=10**(-3), epochs=100):
	a, b = random.random(), random.random()

	n = len(problems)

	rmse_values_per_epoch = []

	for index_epoch in range(epochs):
		a, b, rmse = backpropagation(a, b, n, problems, solutions, learning_rate)
		rmse_values_per_epoch.append(rmse)

	predictions = linear_regression(a, b, problems)

	return a, b, rmse_values_per_epoch, predictions



if __name__ == "__main__":
	house_prices_df = pandas.read_csv("prix_maisons.csv")

	
	print(" Lecture et normalisation ")
	print(house_prices_df.head())
	print(house_prices_df.dtypes)
	print("Nombre de lignes :", len(house_prices_df))

	#  Standardisation
	x_mean, x_std = house_prices_df["surface"].mean(), house_prices_df["surface"].std()
	y_mean, y_std = house_prices_df["prix"].mean(), house_prices_df["prix"].std()

	house_prices_df["surface"] = (house_prices_df["surface"] - x_mean) / x_std
	house_prices_df["prix"] = (house_prices_df["prix"] - y_mean) / y_std

	problems = house_prices_df["surface"]
	solutions = house_prices_df["prix"]

	#  Visualisation
	print("\n Visualisation ")
	plt.figure(figsize=(15, 6))
	plt.plot(problems, solutions, "ro")
	plt.xlabel("surface (standardisee)")
	plt.ylabel("prix (standardise)")
	plt.title("Surface vs Prix")
	plt.show()
    # l'implémentation de notre optimizer
	learning_rate = 0.05
	epochs = 50

	# Entrainement du modele quadratique
	print("\n Modele quadratique — {} epochs ".format(epochs))
	a_q, b_q, c_q, rmse_quadratic = gradient_descent_quadratic(problems, solutions, learning_rate=learning_rate, epochs=epochs)
	print("a={:.4f}, b={:.4f}, c={:.4f}".format(a_q, b_q, c_q))
	print("RMSE finale quadratique :", rmse_quadratic[-1])
	plot_rmse(rmse_quadratic)

	#  Entrainement du modele lineaire
	print("\n Modele lineaire — {} epochs ".format(epochs))
	a_l, b_l, rmse_linear, predictions_linear = gradient_descent(problems, solutions, learning_rate=learning_rate, epochs=epochs)
	print("a={:.4f}, b={:.4f}".format(a_l, b_l))
	print("RMSE finale lineaire :", rmse_linear[-1])

	# Comparaison visuelle
	print("\n Comparaison lineaire vs quadratique ")
	sorted_indices = numpy.argsort(problems)
	sorted_problems = problems.values[sorted_indices]
	predictions_quadratic = quadratic_regression(a_q, b_q, c_q, sorted_problems)
	predictions_lin = linear_regression(a_l, b_l, sorted_problems)
	plot_comparison(sorted_problems, solutions.values[sorted_indices], predictions_lin, predictions_quadratic)

	# Comparaison RMSE
	plt.figure(figsize=(15, 6))
	plt.plot(rmse_linear, "b", label="lineaire")
	plt.plot(rmse_quadratic, "g", label="quadratique")
	plt.xlabel("epochs")
	plt.ylabel("rmse")
	plt.legend()
	plt.title("RMSE lineaire vs quadratique")
	plt.show()
