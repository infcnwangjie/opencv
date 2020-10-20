# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt

#划分袋子区域
def divide_bags(data, radius=50.0):
	clusters = []
	for i in range(len(data)):

		cluster_centroid = data[i]
		cluster_frequency = np.zeros(len(data))
		# Search points in circle
		while True:
			temp_data = []
			for j in range(len(data)):
				v = data[j]

				if np.linalg.norm(v - cluster_centroid) <= radius:
					temp_data.append(v)

					cluster_frequency[i] += 1


			old_centroid = cluster_centroid
			new_centroid = np.average(temp_data, axis=0)
			cluster_centroid = new_centroid
			if np.array_equal(new_centroid, old_centroid):
				break
		# end while

		# Combined 'same' clusters
		has_same_cluster = False
		for cluster in clusters:
			if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= radius:
				has_same_cluster = True
				cluster['frequency'] = cluster['frequency'] + cluster_frequency
				break

		if not has_same_cluster:
			clusters.append({
				'centroid': cluster_centroid,
				'data': [],
				'frequency': cluster_frequency
			})

	# print('clusters (', len(clusters), '): ', clusters)
	clustering(data, clusters)
	# show_clusters(clusters, radius)
	return clusters


# Clustering data using frequency
def clustering(data, clusters):
	frequentlist = []
	for cluster in clusters:
		# cluster['data'] = []
		frequentlist.append(cluster['frequency'])
	frequentarray = np.array(frequentlist)

	# Clustering
	for i in range(len(data)):
		column_frequency = frequentarray[:, i]
		cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
		clusters[cluster_index]['data'].append(data[i])


# Plot clusters
#def show_clusters(clusters, radius):
#	colors = 10 * ['r', 'g', 'b', 'k', 'y']
#	plt.figure(figsize=(5, 5))
#	plt.xlim((0, 500))
#	plt.ylim((0, 500))
#	plt.scatter(X[:, 0], X[:, 1], s=50)
#	theta = np.linspace(0, 2 * np.pi, 800)
#	for i in range(len(clusters)):
#		cluster = clusters[i]
#		data = np.array(cluster['data'])
#		plt.scatter(data[:, 0], data[:, 1], color=colors[i], s=20)
#		centroid = cluster['centroid']
#		plt.scatter(centroid[0], centroid[1], color=colors[i], marker='x', s=30)
#		x, y = np.cos(theta) * radius + centroid[0], np.sin(theta) * radius + centroid[1]
#		plt.plot(x, y, linewidth=1, color=colors[i])
#	plt.show()
##

if __name__ == '__main__':
	X = np.array([
		[450, 320], [432, 318], [480, 319], [463, 310],
		[230, 320], [210, 318],
		[235, 220], [203, 220], [224, 220], [243, 220]
	])

	clusters = divide_bags(X, 60)
	for cluster in clusters:
		print(cluster)
