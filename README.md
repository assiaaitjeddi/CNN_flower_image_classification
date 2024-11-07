<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Augmentation in Flower Classification CNN</title>
</head>
<body>

<h1>Data Augmentation To Address Overfitting In Flower Classification CNN</h1>

<p>This project demonstrates how to build a Convolutional Neural Network (CNN) to classify flower images. We will explore the issue of overfitting in CNNs and address it using data augmentation techniques.</p>

<h2>Table of Contents</h2>
<ol>
    <li><a href="#project-overview">Project Overview</a></li>
    <li><a href="#dependencies">Dependencies</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#model-architecture">Model Architecture</a></li>
    <li><a href="#training-and-evaluation">Training and Evaluation</a></li>
    <li><a href="#data-augmentation">Data Augmentation</a></li>
    <li><a href="#results">Results</a></li>
</ol>

<h2 id="project-overview">Project Overview</h2>
<p>The goal of this project is to classify images of different types of flowers. We will demonstrate how to train a CNN to classify the flower images and then explore how data augmentation can help reduce overfitting by generating new training samples through transformations like rotations, zoom, and flips.</p>

<h2 id="dependencies">Dependencies</h2>
<p>Make sure to install the following Python libraries:</p>
<ul>
    <li>TensorFlow</li>
    <li>NumPy</li>
    <li>OpenCV</li>
    <li>Matplotlib</li>
    <li>Pillow</li>
</ul>
<p>You can install them using pip:</p>
<pre><code>pip install tensorflow numpy opencv-python matplotlib pillow</code></pre>

<h2 id="dataset">Dataset</h2>
<p>This project uses the <strong>Flowers Dataset</strong> available from TensorFlow. It contains images of 5 different flower types: roses, daisies, dandelions, sunflowers, and tulips.</p>
<p>You can download the dataset using the following command:</p>
<pre><code>dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
</code></pre>
<p>The dataset consists of 3,670 images, with approximately 700 images per class.</p>

<h2 id="model-architecture">Model Architecture</h2>
<p>We build a simple CNN architecture with the following layers:</p>
<ol>
    <li>Conv2D with 16 filters</li>
    <li>MaxPooling2D</li>
    <li>Conv2D with 32 filters</li>
    <li>MaxPooling2D</li>
    <li>Conv2D with 64 filters</li>
    <li>MaxPooling2D</li>
    <li>Flatten layer</li>
    <li>Dense layer with 128 units</li>
    <li>Output layer with 5 units (for 5 flower classes)</li>
</ol>
<p>We compile the model with the Adam optimizer and Sparse Categorical Crossentropy loss function.</p>

<h2 id="training-and-evaluation">Training and Evaluation</h2>
<p>The model is trained for 30 epochs on the flower dataset. We use <code>train_test_split</code> to split the data into training and testing sets, followed by image normalization (scaling the pixel values to the range [0, 1]).</p>
<p>After training the model, we evaluate it on the test data. The initial results may show a high accuracy on the training set but lower accuracy on the test set, indicating overfitting.</p>

<h2 id="data-augmentation">Data Augmentation</h2>
<p>To address the overfitting issue, we apply <strong>data augmentation</strong>. This technique generates new training samples using transformations such as:</p>
<ul>
    <li>Horizontal flips</li>
    <li>Random rotations</li>
    <li>Random zoom</li>
</ul>
<p>We define the data augmentation as follows:</p>
<pre><code>data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])
</code></pre>
<p>This augmentation is applied directly in the model as the first layer before the CNN layers.</p>

<h2 id="results">Results</h2>
<p>After training the model with data augmentation, we observe an improvement in the model’s generalization performance on the test set. The model's accuracy on the test set increases, demonstrating that data augmentation has helped reduce overfitting.</p>

<h3>Example Predictions</h3>
<ul>
    <li><strong>Before data augmentation:</strong> The model shows high training accuracy but low test accuracy, suggesting overfitting.</li>
    <li><strong>After applying data augmentation:</strong> The model's test accuracy improves, and the model performs better on unseen data.</li>
</ul>

<h2>Conclusion</h2>
<p>This project illustrates how to apply data augmentation to mitigate overfitting in CNN models. By using transformations on the training data, the model is able to generalize better and perform well on unseen test data.</p>

<h2>Credits</h2>
<p>The code is based on the official TensorFlow tutorial <a href="https://www.tensorflow.org/tutorials/images/classification">Image Classification</a> with modifications for simplicity.</p>

</body>
