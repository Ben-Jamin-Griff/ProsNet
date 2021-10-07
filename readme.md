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

This respository contains the code and models described in the publication:

*A Machine Learning Classification Model for Monitoring the Daily Physical Behaviour of Lower-Limb Amputees" (currently under review)*.

Currently this code works with data export from the activPAL activtiy monitor:

[palt.com](https://www.palt.com/)

Here are the main uses of this software:
* Process shank activPAL accelerometer data to estimate physical behaviour postures
* Process shank activPAL accelerometer data with thigh activPAL event data for creating labeled datasets
* Re-create the model development process used in Griffiths et al. (2021)
* Experiment with new model development methods TBA
* Estimate non-wear periods from activPAL accelerometer data - algorithm validation ongoing

This repository is constantly being updated. Check back for more info...

### Built With

* 🐍 [Python](https://www.python.org)
* [Scikit-Learn](https://www.scikit-learn.org)

<!-- GETTING STARTED -->
## 🚀 Getting Started

To get a local copy up and running follow these simple example steps.

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
   pip install -r requirements.txt
   ```

## Usage

Make sure you have cloned the repository and installed `requirements.txt`

Just run the following command at the root of your project:

```sh
python3 examples/
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

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

TBC
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
