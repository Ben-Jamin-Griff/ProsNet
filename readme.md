<h1 align="center">ProsNet</h1>
<p align="center">
  A software package for developing classification models that predict physical behaviour postures.
  <br />
  <a href="https://github.com/Ben-Jamin-Griff/ProsNet"><strong>Explore the docs »</strong></a>
  <br />
  <br />
</p>

## 🤔 About The Project

<!--[![Product Name Screen Shot][product-screenshot]](https://example.com)-->

This respository contains the sotware package and models described in the publication:

*A Machine Learning Classification Model for Monitoring the Daily Physical Behaviour of Lower-Limb Amputees" (Griffiths et al., 2021)*.

The code works with data export from the activPAL activtiy monitor [palt.com](https://www.palt.com/)

Here are the main uses for this software:

* Estimate physical behaviour postures from shank accelerometer data
* Process shank accelerometer data along with thigh accelerometer event data to create a labeled dataset for training:
  * Machine learning classifiers from heuristic features
  * Deep learning classifiers from windowed acceleration data
* Re-create the model development process used in Griffiths et al. (2021)
* Experiment with new model development
* Estimate non-wear periods from accelerometer data

See the [example scripts](https://github.com/Ben-Jamin-Griff/ProsNet/tree/main/examples) for each of these use cases.

### Built With

* 🐍 [Python](https://www.python.org)
* [Scikit-Learn](https://www.scikit-learn.org)
* [TensorFlow](https://www.tensorflow.org/)


## 🚀 Getting Started

Test out the package and start processing data.

### 💻 Prerequisites

You need these pre-installed on your device to get started.

* Python & pip: A useful resource for installing python - [instructions](https://realpython.com/installing-python/)

### Installation

1. Open your terminal/shell and navigate to the directory where you want to install this software
2. Clone the repo
   ```sh
   git clone https://github.com/Ben-Jamin-Griff/ProsNet.git
   ```
3. Move into repo
   ```sh
   cd ProsNet
   ```  
4. Install Python packages
   ```sh
   pip install ProsNet
   ```

### Usage

Make sure you completed the installation steps and then run the following command:

* Unix/maxOS 
```sh
python3 examples/shallow_examples/example_1.py
```

* Windows
```sh
py examples\shallow_examples\example_1.py
```

## 🗺️ Exploring The Package

To get a local copy up and running follow these simple steps.

### Installation

1. Open your terminal/shell and navigate to the directory where you want to install this software
2. Clone the repo
   ```sh
   git clone https://github.com/Ben-Jamin-Griff/ProsNet.git
   ```
3. Move into repo
   ```sh
   cd ProsNet
   ```  
4. Install Python packages
   ```sh
   pip install -r requirements.txt
   ```

## 🤝 Contributing

Contributions are what make the open source community such an amazing place. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- Author -->
## Author

👤 **Benjamin Griffiths**

- [Twitter](https://twitter.com/ben_jamin_griff)
- [Github](https://github.com/Ben-Jamin-Griff)

<!-- ACKNOWLEDGEMENTS
## Acknowledgements

#TBC -->
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
-->
