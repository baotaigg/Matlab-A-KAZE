# AKAZE Full (MATLAB Implementation)

This is a simplified **AKAZE** feature extraction pipeline implemented in MATLAB. It performs multiscale feature detection using **Perona-Malik nonlinear diffusion**, Hessian-based corner detection, and a handcrafted binary descriptor with orientation alignment.

## 🧠 Features

- Nonlinear scale space construction via **Perona-Malik diffusion**
- Feature point detection using **Hessian response extrema**
- Orientation assignment based on local gradients
- Binary descriptor generation using **LDB-style patch encoding**
- Pure MATLAB implementation, no external toolbox required

## 📁 File Structure

| Function | Description |
|----------|-------------|
| `akaze_full.m` | Main function for detecting keypoints and computing descriptors |
| `test.m` | Test main function|

## 🧪 Usage

```matlab
img = imread('example.jpg');
[keypoints, descriptors, responseMaps] = akaze_full(img);
