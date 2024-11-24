# Get predictions on test images
predictions = model.predict(test_data_gen, steps=test_data_gen.samples)
predicted_class_indices = np.round(predictions).astype(int)

# Display results
def plot_images(images, predictions, labels):
    for i in range(len(images)):
        plt.imshow(images[i])
        plt.title(f'Predicted: {"Dog" if predictions[i] == 1 else "Cat"}')
        plt.show()

test_images, _ = next(test_data_gen)
plot_images(test_images, predicted_class_indices, test_data_gen.labels)
